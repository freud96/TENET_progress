# Introduction

This document studies MLC-LLM's, pipeline.py, matmul.py of Dlight. The former is to compile the model into executable library to deploy on native hardware, the latter is written as a pass to optimize IRModule, GEMM schefule rule for GPU operators. This documents mainly has 3 parts:

1. Detailing pipeline.py, when and how it is called
2. How the IRModule, which block it targets.
3. Case study of GPT-2

## Objective

The primary objective is to understand how MLC-LLM compiles a model into its corresponding library, and how to retrieve workload dimension of matrix multiplication, which is the bottleneck of LLM, during runtime in order to estimate the cost of some algorithm techniques on hardware accelerators(TENET-like).

## Compilation process

In order to generate the library.so file for specific device to deploy a chosen model, there are a set of commands that needs to be run. convert_weight, gen_config and compile.

* convert_weight: Command to convert weight into MLC format. It can also quantizes the weights
* gen_config: To generate mlc-chat-config.json. File that contains specifications that can alter the compiled model. Tokenizers are also processed in this step.
* compile: Compile the model into a model library(.so, .tar, etc) containing the inference logic of a model.

For the compile process, compile.py is used. The dataclass CompileArgs encapsulates the arguments required for the compilation process. It includes:

* _config_: the model configuration path.
* _quantization_: Details about the quantization method being used.
* _model_: The model to be compiled.
* _target_: The target hardware to which the model is being compiled(CUDA, CPU).
* _opt_: Optimization flags specific to the target hardware.
* _build_func_ : A function that performs the build process, taking the IRModule and CompileArgs as input.

The main function for compiling the model perform the following steps below:

* Step 1: Model configuration, it applies any overrides to the model configuration and enables specific operators based on the target hardware and optimization flags.

    ```python
    model_config = args.overrides.apply(model_config)
    ```

* Step 2: Create the model from the configuration given in the .json file(vocab size, n_embd, n_layer, n_head, layer_norm_epsilon, n_unk_size, etc) apply quantization if required.

    ``` python
    logger.info("Creating model from: %s", args.config)
            if (
                args.quantization.kind == "ft-quant"
                and hasattr(model_config, "tensor_parallel_shards")
                and model_config.tensor_parallel_shards > 1  # type: ignore
            ):
                raise NotImplementedError
            if (
                hasattr(args.quantization, "linear_weight_layout")
                and args.quantization.linear_weight_layout == "KN"
                and hasattr(model_config, "tensor_parallel_shards")
                and model_config.tensor_parallel_shards > 1  # type: ignore
            ):
                raise NotImplementedError(
                    "KN layout (q3f16_0 and q4f16_0) is not supported for tensor parallelism"
                )
            model, _ = args.model.quantize[args.quantization.kind](model_config, args.quantization)
    ```

* Step 3: Export the model to TVM Unity, the backend compiler for MLC-LLM, the IRModule is then generated

    ```python
    logger.info("Exporting the model to TVM Unity compiler")
            mod, named_params, ext_mods = model.export_tvm(
                spec=model.get_default_spec(),  # type: ignore
                allow_extern=True,
            )
    ```

* Step 4: Run optimizations using TVM unity. In this step, preprocessing are applied to the model parameters, metadata about the model is registered, which is crucial for debugging purpose and optimizing during model execution.

    ``` python
    logger.info("Running optimizations using TVM Unity")
    additional_tirs = _apply_preproc_to_params_and_check_pipeline(named_params, model_config)
    variable_bounds = _get_variable_bounds(model_config)
    cuda_graph_symbolic_capture_hints = {
        "batch_decode": ["batch_size"],
        "batch_decode_to_last_hidden_states": ["batch_size"],
        "batch_verify": ["batch_size", "seq_len"],
        "batch_verify_to_last_hidden_states": ["batch_size", "seq_len"],
    }
    metadata = {
        "model_type": args.model.name,
        "quantization": args.quantization.name,
        "context_window_size": getattr(model_config, "context_window_size", -1),
        "sliding_window_size": getattr(model_config, "sliding_window_size", -1),
        "attention_sink_size": getattr(model_config, "attention_sink_size", -1),
        "prefill_chunk_size": model_config.prefill_chunk_size,  # type: ignore
        "tensor_parallel_shards": model_config.tensor_parallel_shards,  # type: ignore
        "pipeline_parallel_stages": getattr(model_config, "pipeline_parallel_stages", 1),
        "kv_state_kind": _infer_kv_state_kind(args.model.name),
        "max_batch_size": getattr(model_config, "max_batch_size", 1),
    }
    logger.info("Registering metadata: %s", metadata)
    metadata["params"] = [_get_param_metadata(name, param) for name, param in named_params]
    with PassContext(config={"relax.backend.use_cuda_graph": args.opt.cudagraph}):
        args.build_func(
            mod,
            args,
            pipeline=relax.get_pipeline(  # type: ignore
                "mlc_llm",
                target=args.target,
                flashinfer=args.opt.flashinfer,
                cublas_gemm=args.opt.cublas_gemm,
                faster_transformer=args.opt.faster_transformer,
                allreduce_strategy=args.opt.ipc_allreduce_strategy,
                variable_bounds=variable_bounds,
                cuda_graph_symbolic_capture_hints=cuda_graph_symbolic_capture_hints,
                additional_tirs=additional_tirs,
                ext_mods=ext_mods,
                metadata=metadata,
                debug_dump=args.debug_dump,
            ),
        )
    ```

    The generated IRModule is gone through a series of transform passes using TVM's PassContext to run the compilation pipeline with the specified optimization and debugging settings. The transformation sequence is organized into multiple phases, each perform specific tasks to optimize and prepare the moddel for execution.

  * Phase 0: Passes that prepare the model by adding metadata, handling tensor parallelism, setting up pipeline parallelism and optimizing for GPU execution. In this phase, unused relax functions are also removed.

    ```python
        DispatchKVCacheCreation(target, flashinfer, metadata),
        AttachSoftmaxWithTemperature(target),
        AttachVariableBounds(variable_bounds),
        AttachCUDAGraphSymbolicCaptureHints(cuda_graph_symbolic_capture_hints),
        AttachPipelineParallelStages(metadata["pipeline_parallel_stages"]),
        AttachLogitProcessFunc(target),
        AttachAdditionalPrimFuncs(additional_tirs),
        AttachAllocEmbeddingTensorFunc(metadata),
        AttachGPUSamplingFunc(target, variable_bounds),
        AttachSpecDecodeAuxFuncs(tensor_parallel_shards),
        AttachMemoryPlanAttr(),
        tvm.tir.transform.BindTarget(tvm.target.Target.current(allow_none=False)),
        _DebugDump("debug-phase0.py", debug_dump, show_meta=False),
    ```

  * Phase 1: In this phase, some TVM Relax optimizations are run, for example, fusing Tranpose matrix multiplications, fusing adds and RMS norm, as well as dequantize etc.
  
    ```python
        FuseFTDequantizeEpilogue(),
        FuseDequantizeTranspose(),
        CublasDispatch() if cublas_gemm else tvm.transform.Sequential([]),
        FuseAddRMSNorm(target=target),
        FuseTransposeMatmul(),
        _DebugDump("debug-phase1.py", debug_dump, show_meta=False),
    ```

  * Phase 2: Lowering to TVM TIR kernels, in this phase, first the operations are legalized, then they are fused with transform.FuseOps and then the operations are lowered to TensorIR.
  
    ```python
        tvm.relax.backend.DispatchSampling(),
        tvm.relax.backend.DispatchSortScan(),
        tvm.relax.transform.LegalizeOps(),
        tvm.relax.transform.AnnotateTIROpPattern(),
        tvm.relax.transform.FoldConstant(),
        tvm.relax.transform.FuseOps(),
        tvm.relax.transform.FuseTIR(),
        _DebugDump("debug-phase2.py", debug_dump, show_meta=False),
    ```

  * Phase 3: Optimizations at TIR-level is done, like fusing matrix multiplaction and dequantization, and removing dead codes.
  
    ```python
        FuseDequantizeMatmulEwise(),
        FuseDequantizeTake(),
        tvm.relax.transform.DeadCodeElimination(),
        CleanUpTIRAttrs(["op_pattern"]),
        _DebugDump("debug-phase3.py", debug_dump, show_meta=False),
    ```

  * Phase 4: At this phase, there TVM Dlight low-level optimizations are done, like scheduling matrix-multiplication, matrix-vector multiplication for GPU, as well as GPU reduction and fallback.
  
    ```python
        LowBatchGemvSpecialize(),
        dl.ApplyDefaultSchedule(
        dl.gpu.Matmul(),
        dl.gpu.GEMV(),
        dl.gpu.Reduction(),
        dl.gpu.GeneralReduction(),
        dl.gpu.Fallback(),
        ),
        _DebugDump("debug-phase4.py", debug_dump, show_meta=False),
    ```

  * Phase 5: Lowering to VM bytecode. In this phase, before lowering to VM bytecode, many things happen; For instance, rewriting the dataflow, rewrite CUDA graph, tensor allocations etc.
    
    ```python
        LiftTIRGlobalBufferAlloc(),
        (
            tvm.tir.transform.ForceNarrowIndexToInt32()
            if target.kind.name != "cuda"
            else tvm.transform.Sequential([])
        ),
        ScatterTupleGetItem(),
        tvm.relax.transform.RewriteDataflowReshape(),
        tvm.relax.transform.ToNonDataflow(),
        tvm.relax.transform.RemovePurityChecking(),
        tvm.relax.transform.CallTIRRewrite(),
        (
            tvm.relax.transform.IPCAllReduceRewrite(allreduce_strategy)
            if allreduce_strategy != IPCAllReduceStrategyType.NONE
            else tvm.transform.Sequential([])
        ),
        tvm.relax.transform.StaticPlanBlockMemory(),
        AttachMetadataWithMemoryUsage(metadata),
        tvm.relax.transform.RewriteCUDAGraph(),
        tvm.relax.transform.LowerGPUIPCAllocStorage(),
        tvm.relax.transform.LowerAllocTensor(),
        tvm.relax.transform.KillAfterLastUse(),
        tvm.relax.transform.VMBuiltinLower(),
        tvm.relax.transform.VMShapeLower(),
        tvm.relax.transform.AttachGlobalSymbol(),
        _DebugDump("debug-final.py", debug_dump, show_meta=False),
    ```

  N.B: we do not cover the detail of each functions in this document  


## Code Tracing of the GPU schedule rule

In phase 4 of the compilation pipeline, MLC LLM calls on TVM's dlight to schedule the module for GPU, matrix multiplication, matrix-vector multiplication, reduction etc. 

Our main focus will be on matrix multiplication schedule rule; In the python script, there are 3 classes that can apply schedule rule depending on the input/output data type, for instance, rule for float16 and rule for int8 tensor cores

The general GPU schedule rule class is as follow:

```python 
class Matmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and "android" in str(target.host):
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=64,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        index_maps = get_index_map(block_stmt)
        if index_maps is None:
            return None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        block = sch.reindex(main_block, ("read", 0))
        sch.transform_layout(block, ("write", 0), a_index_map)
        block = sch.reindex(main_block, ("read", 1))
        sch.transform_layout(block, ("write", 0), b_index_map)
        block = sch.reindex(main_block, ("write", 0))
        sch.transform_layout(block, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent, tir.expr.IntImm):
                    if extent.value <= minimal_tensorize_threshold:
                        apply_tensorization = False
            if apply_tensorization:
                # Analyze read/write buffers and choose correct tensorizer: int8 or fp16.
                in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
                tensorize_sch = None
                if in_dtype == "int8" and out_dtype == "int32":
                    tensorize_sch = MatmulInt8Tensorization().apply(func, target, _)
                elif in_dtype == "float16" and out_dtype in ["float16", "float32"]:
                    tensorize_sch = MatmulTensorization().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch

        # Step 2. Get schedule config.
        config = self.get_configs(target)

        # Step 3. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        reordered_loops = [by, bx, vy, vx, ty, tx, ko, ki] + (
            [yi, xi] if config.inner_x else [xi, yi]
        )
        sch.reorder(*reordered_loops)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(reordered_loops[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)

        sch.decompose_reduction(main_block, ko)
        return sch

```

The 
>>>>>>> adeb664 (add log_data.cc, matmul1.py and try fash)
