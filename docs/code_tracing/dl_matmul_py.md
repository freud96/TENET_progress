# Introduction

This document studies MLC-LLM's, pipeline.py, matmul.py of Dlight. The former is to compile the model into executable library to deploy on native hardware, the latter is written as a pass to optimize IRModule, GEMM schefule rule for GPU operators. This documents mainly has 3 parts:

1. Detailing pipeline.py, when and how it is called
2. How the IRModule, whioch block it targets.
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

## Extracting the workload dimensions during runtime
