# Adding Hooks to IRModule to capture parameters, variables, constant during runtime

In this document, our goal is to define and describe the process to extract workload dimensions for Matrix multiplication in order to profile LLM on hardware accelerator. To recap, below table summarizes the phases an IRModule goes through during compilation process.

<table>
  <tr>
    <th style="background-color: #ADD8E6;"></th>
    <th style="background-color: #B0E0E6;">Phases 0</th>
    <th style="background-color: #ADD8E6;">Phases 1</th>
    <th style="background-color: #B0E0E6;">Phases 2</th>
    <th style="background-color: #ADD8E6;">Phases 3</th>
    <th style="background-color: #B0E0E6;">Phases 4</th>
    <th style="background-color: #ADD8E6;">Phases 5</th>

  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Functions</td>
    <td style="background-color: #F0FFFF; text-align: center;"> Adding metadata, 
    Handling Tensor parallelism, Remove unused relax functions</td>
    <td style="background-color: #E0FFFF; text-align: center;">Perform Relax optimizations,
    (Like fusing transpose matrix multiplication, adds operations, RMS norms and dequantize)</td>
    <td style="background-color: #F0FFFF; text-align: center;">Lower the IRModule relax functions to TVM TIR kernels, operations are legalized, then fused</td>
    <td style="background-color: #E0FFFF; text-align: center;">Optimizations at TIR-level, fusing matrix multiplication element wise and dequantization, remove dead codes</td>
    <td style="background-color: #F0FFFF; text-align: center;">Perform low-level optimizations, schedule matrix-multiplication, matrix-vector multiplication for GPU</td>
    <td style="background-color: #E0FFFF; text-align: center;">Lower to VM bytecode, rewrite CUDA graph, tensor alloacations, rewrite dataflow</td>
  </tr>

</table>

## Examining a function right after being lowered to TIR and its derived GPU-optimized

Considering one TIR function after phase 3 of compiling process of GPT-2. This function contains 5 compute blocks, namely compute, dequantize, NT_matmul, T_add and T_add_1. As the names already give a hint, except for compute which is for .."to be added".

from debug-phase3.py

```python
    def fused_fused_dequantize1_fused_NT_matmul6_add7_add8(lv397: T.Buffer((T.int64(1024), T.int64(128)), "uint32"), lv398: T.Buffer((T.int64(1024), T.int64(32)), "float16"), p_reshape195: T.handle, transformer_h_0_attn_c_proj_bias3: T.Buffer((T.int64(1024),), "float16"), p_add290: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        reshape195 = T.match_buffer(p_reshape195, (T.int64(1), seq_len, T.int64(1024)), "float16")
        add290 = T.match_buffer(p_add290, (T.int64(1), seq_len, T.int64(1024)), "float16")
        T_add_intermediate_1_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(1024)), "float16")
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
        for i0, i1 in T.grid(T.int64(1024), T.int64(1024)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv397[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv397[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(1024), T.int64(1024)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], lv398[v_i0, v_i1 // T.int64(32)])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv398[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(1024), T.int64(1024)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(reshape195[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + reshape195[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1024)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(NT_matmul_intermediate[v_ax0, v_ax1, v_ax2], transformer_h_0_attn_c_proj_bias3[v_ax2])
                T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate[v_ax0, v_ax1, v_ax2] = NT_matmul_intermediate[v_ax0, v_ax1, v_ax2] + transformer_h_0_attn_c_proj_bias3[v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), seq_len, T.int64(1024)):
            with T.block("T_add_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2], add290[v_ax0, v_ax1, v_ax2])
                T.writes(T_add_intermediate_1_intermediate[v_ax0, v_ax1, v_ax2])
                T_add_intermediate_1_intermediate[v_ax0, v_ax1, v_ax2] = T_add_intermediate[v_ax0, v_ax1, v_ax2] + add290[v_ax0, v_ax1, v_ax2]
```

In order for TVM to support dynamic shapes, relax was introduced. relax gives us the capability to give input shape during runtime. For example in this module above, just like any other TIR fucntion, it has 3 parts: the buffers(input, output and intermediate results), the loop nests(multiple block of loop nests sometimes interlacing, for multiple blocks) and computations statements.

<table>
  <tr>
    <th style="background-color: #ADD8E6;">Parameters</th>
    <th style="background-color: #B0E0E6;">Type</th>
    <th style="background-color: #ADD8E6;">Size</th>
    <th style="background-color: #B0E0E6;">I/O/Constant</th>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">lv397</td>
    <td style="background-color: #F0FFFF; text-align: center;">buffer</td>
    <td style="background-color: #E0FFFF; text-align: center;">(1024, 128)</td>
    <td style="background-color: #F0FFFF; text-align: center;">input</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">lv398</td>
    <td style="background-color: #F0FFFF; text-align: center;">buffer</td>
    <td style="background-color: #E0FFFF; text-align: center;">(1024, 32)</td>
    <td style="background-color: #F0FFFF; text-align: center;">input</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">p_reshape195</td>
    <td style="background-color: #F0FFFF; text-align: center;">handle</td>
    <td style="background-color: #E0FFFF; text-align: center;">*</td>
    <td style="background-color: #F0FFFF; text-align: center;">input</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">transformer_h_0_attn_c_proj_bias3</td>
    <td style="background-color: #F0FFFF; text-align: center;">buffer</td>
    <td style="background-color: #E0FFFF; text-align: center;">(1024, )</td>
    <td style="background-color: #F0FFFF; text-align: center;">input</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">p_add290</td>
    <td style="background-color: #F0FFFF; text-align: center;">handle</td>
    <td style="background-color: #E0FFFF; text-align: center;">*</td>
    <td style="background-color: #F0FFFF; text-align: center;">input</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">p_output0</td>
    <td style="background-color: #F0FFFF; text-align: center;">handle</td>
    <td style="background-color: #E0FFFF; text-align: center;">*</td>
    <td style="background-color: #F0FFFF; text-align: center;">output</td>
  </tr>
</table>

In the table above, there are two types: T.buffer and T.handle. T.buffer is used to describe a tensor with explicit shape and data type information. Elements can be accessed directly and the shape is assigned in memory during compilation. In contrast, T.handle is a more abstract representation of memory. It does not contain any information about the tensor(nor shape nor data type). it makes a tensor flexible and will be more concrete at runtime. 

the dynamic shapes of the tensors in the above module depends on seq_len(sequence length). As seen in the codes below, where the variables(input or local variables) are declared. 

```python
    seq_len = T.int64()
    reshape195 = T.match_buffer(p_reshape195, (T.int64(1), seq_len, T.int64(1024)), "float16")
    add290 = T.match_buffer(p_add290, (T.int64(1), seq_len, T.int64(1024)), "float16")
    T_add_intermediate_1_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(1024)), "float16")
    # with T.block("root"):
    compute = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
    dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
    NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
    T_add_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
```

The seq_len is declared without passing a number inside the parentheses. This marks that it will be allocated, reallocated during runtime. The variables like reshapes195, add290, from the input parameters are declared as T.match_buffer, this type is just like T.buffer with the primary difference of reintepreting a handle(dynamic) as specific buffer while T.buffer is for static.

The rest of variables depending on the dynamic input seq_len are local variables as well as the output tensor.

From debug-final.py

```python
    @T.prim_func(private=True)
    def fused_fused_dequantize1_fused_NT_matmul6_add7_add8(lv397: T.Buffer((T.int64(1024), T.int64(128)), "uint32"), lv398: T.Buffer((T.int64(1024), T.int64(32)), "float16"), p_reshape195: T.handle, transformer_h_0_attn_c_proj_bias3: T.Buffer((T.int64(1024),), "float16"), p_add290: T.handle, p_output0: T.handle):
        T.func_attr({"tir.HoistIfThenElseExprWithBlock": 1, "tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
        seq_len = T.int64()
        reshape195 = T.match_buffer(p_reshape195, (T.int64(1), seq_len, T.int64(1024)), "float16")
        add290 = T.match_buffer(p_add290, (T.int64(1), seq_len, T.int64(1024)), "float16")
        T_add_intermediate_1_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(1024)), "float16")
        # with T.block("root"):
        if T.tvm_thread_invariant(seq_len <= T.int64(2)):
            with T.block("root"):
                T.reads()
                T.writes()
                dequantize_intermediate_intermediate_local = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16", scope="local")
                NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(1)) // T.int64(2) * T.int64(2), T.int64(1024)), "float16", scope="local")
                NT_matmul_intermediate_pad_rf_local = T.alloc_buffer((T.int64(128), T.int64(1), (seq_len + T.int64(1)) // T.int64(2) * T.int64(2), T.int64(1024)), "float16", scope="local")
                NT_matmul_intermediate_pad_rf_local_1 = T.alloc_buffer((T.int64(32), T.int64(1), (seq_len + T.int64(1)) // T.int64(2) * T.int64(2), T.int64(1024)), "float16", scope="local")
                for ax0_0 in T.thread_binding((seq_len + T.int64(1)) // T.int64(2), thread="blockIdx.y"):
                    for u_fused_ax1_fused_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
                        for u_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(2), T.int64(2)):
                                    for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                        with T.block("NT_matmul_rf_init"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                            v0 = T.axis.spatial((seq_len + T.int64(1)) // T.int64(2) * T.int64(2), ax0_0 * T.int64(2) + ax0_1_init)
                                            v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                            T.reads()
                                            T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1])
                                            NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] = T.float16(0)
                                for ax2_fused_u_fused_0 in T.serial(T.int64(4), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                    for ax0_0_1, ax1 in T.grid(T.int64(2), T.int64(8)):
                                        for ax0_1 in T.vectorized(T.int64(1)):
                                            with T.block("dequantize"):
                                                v0 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + ax0_0_1 + ax0_1)
                                                v1 = T.axis.spatial(T.int64(1024), ax2_fused_u_fused_0 * T.int64(256) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(8) + ax1)
                                                T.reads(lv397[v0, v1 // T.int64(8)], lv398[v0, v1 // T.int64(32)])
                                                T.writes(dequantize_intermediate_intermediate_local[v0, v1])
                                                dequantize_intermediate_intermediate_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv397[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv398[v0, v1 // T.int64(32)]
                                    for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(2), T.int64(2), T.int64(2)):
                                        for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                            with T.block("NT_matmul_rf_update"):
                                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                                v0 = T.axis.spatial((seq_len + T.int64(1)) // T.int64(2) * T.int64(2), ax0_0 * T.int64(2) + ax0_1)
                                                v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                                vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                                T.reads(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1], reshape195[T.int64(0), v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                                T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1])
                                                NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] = NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] + T.if_then_else(v0 < seq_len, reshape195[T.int64(0), v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                        for ax3_fused_0_ax3_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                    for ax2 in range(T.int64(2)):
                                        for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                            with T.block("NT_matmul_rf_init"):
                                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                                v0 = T.axis.spatial((seq_len + T.int64(1)) // T.int64(2) * T.int64(2), ax0_0 * T.int64(2) + ax2)
                                                v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                                T.reads()
                                                T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                                NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] = T.float16(0)
                                            for ax1 in range(T.int64(4)):
                                                with T.block("NT_matmul_rf_update"):
                                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                                    v0 = T.axis.spatial((seq_len + T.int64(1)) // T.int64(2) * T.int64(2), ax0_0 * T.int64(2) + ax2)
                                                    v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                                    T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1], NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, v1])
                                                    T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                                    NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] = NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] + NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, v1]
                        for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(2)):
                            for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    with T.block("NT_matmul"):
                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                                        v0 = T.axis.spatial((seq_len + T.int64(1)) // T.int64(2) * T.int64(2), ax0_0 * T.int64(2) + ax1)
                                        v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                        T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                        T.writes(NT_matmul_intermediate_pad_local[T.int64(0), v0, v1])
                                        with T.init():
                                            NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] = T.float16(0)
                                        NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] = NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] + NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1]
                        for ax0 in range(T.int64(2)):
                            for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax1_fused_2 in range(T.int64(2)):
                                    with T.block("NT_matmul_intermediate_pad"):
                                        v0 = T.axis.spatial(seq_len, ax0_0 * T.int64(2) + ax0)
                                        v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                        T.where((ax0_0 - (seq_len + T.int64(1)) // T.int64(2) < T.int64(0) or ax0_0 == T.int64(0)) and ax0_0 * T.int64(2) + ax0 < seq_len)
                                        T.reads(NT_matmul_intermediate_pad_local[T.int64(0), v0, v1], transformer_h_0_attn_c_proj_bias3[v1], add290[T.int64(0), v0, v1])
                                        T.writes(T_add_intermediate_1_intermediate[T.int64(0), v0, v1])
                                        T_add_intermediate_1_intermediate[T.int64(0), v0, v1] = NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] + transformer_h_0_attn_c_proj_bias3[v1] + add290[T.int64(0), v0, v1]
        else:
            if T.tvm_thread_invariant(seq_len <= T.int64(8)):
                with T.block("root"):
                    T.reads()
                    T.writes()
                    dequantize_intermediate_intermediate_local = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16", scope="local")
                    NT_matmul_intermediate_pad_local = T.alloc_buffer((T.int64(1), (seq_len + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1024)), "float16", scope="local")
                    NT_matmul_intermediate_pad_rf_local = T.alloc_buffer((T.int64(128), T.int64(1), (seq_len + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1024)), "float16", scope="local")
                    NT_matmul_intermediate_pad_rf_local_1 = T.alloc_buffer((T.int64(32), T.int64(1), (seq_len + T.int64(3)) // T.int64(4) * T.int64(4), T.int64(1024)), "float16", scope="local")
                    for ax0_0 in T.thread_binding((seq_len + T.int64(3)) // T.int64(4), thread="blockIdx.y"):
                        for u_fused_ax1_fused_fused_0 in T.thread_binding(T.int64(32), thread="blockIdx.x"):
                            for u_fused_ax1_fused_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_1_init, u_fused_ax1_fused_fused_2_init in T.grid(T.int64(4), T.int64(2)):
                                        for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                                            with T.block("NT_matmul_rf_init"):
                                                vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1_init)
                                                v0 = T.axis.spatial((seq_len + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1_init)
                                                v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2_init)
                                                T.reads()
                                                T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1])
                                                NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] = T.float16(0)
                                    for ax2_fused_u_fused_0 in T.serial(T.int64(4), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                        for ax0_0_1, ax1 in T.grid(T.int64(2), T.int64(8)):
                                            for ax0_1 in T.vectorized(T.int64(1)):
                                                with T.block("dequantize"):
                                                    v0 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + ax0_0_1 + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(1024), ax2_fused_u_fused_0 * T.int64(256) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(8) + ax1)
                                                    T.reads(lv397[v0, v1 // T.int64(8)], lv398[v0, v1 // T.int64(32)])
                                                    T.writes(dequantize_intermediate_intermediate_local[v0, v1])
                                                    dequantize_intermediate_intermediate_local[v0, v1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv397[v0, v1 // T.int64(8)], T.Cast("uint32", v1 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv398[v0, v1 // T.int64(32)]
                                        for ax0_1, u_fused_ax1_fused_fused_2, ax2_fused_u_fused_2 in T.grid(T.int64(4), T.int64(2), T.int64(2)):
                                            for ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 in T.vectorized(T.int64(4)):
                                                with T.block("NT_matmul_rf_update"):
                                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused = T.axis.spatial(T.int64(128), ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + ax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1)
                                                    v0 = T.axis.spatial((seq_len + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax0_1)
                                                    v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + u_fused_ax1_fused_fused_1 * T.int64(2) + u_fused_ax1_fused_fused_2)
                                                    vax2_fused_u_fused_0, vax2_fused_u_fused_2 = T.axis.remap("RR", [ax2_fused_u_fused_0, ax2_fused_u_fused_2])
                                                    T.reads(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1], reshape195[T.int64(0), v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)])
                                                    T.writes(NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1])
                                                    NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] = NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused, T.int64(0), v0, v1] + T.if_then_else(v0 < seq_len, reshape195[T.int64(0), v0, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)], T.float16(0)) * dequantize_intermediate_intermediate_local[v1, vax2_fused_u_fused_0 * T.int64(256) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused // T.int64(4) * T.int64(8) + vax2_fused_u_fused_2 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused % T.int64(4)]
                            for ax3_fused_0_ax3_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax3_fused_2_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                                        for ax2 in range(T.int64(4)):
                                            for ax3_fused_2_1 in T.vectorized(T.int64(2)):
                                                with T.block("NT_matmul_rf_init"):
                                                    vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                                                    v0 = T.axis.spatial((seq_len + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                                    v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                                    T.reads()
                                                    T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                                    NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] = T.float16(0)
                                                for ax1 in range(T.int64(4)):
                                                    with T.block("NT_matmul_rf_update"):
                                                        vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                                        v0 = T.axis.spatial((seq_len + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax2)
                                                        v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax3_fused_0_ax3_fused_1_fused * T.int64(2) + ax3_fused_2_0 * T.int64(2) + ax3_fused_2_1)
                                                        T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1], NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, v1])
                                                        T.writes(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                                        NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] = NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1] + NT_matmul_intermediate_pad_rf_local[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 * T.int64(4) + vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_1, T.int64(0), v0, v1]
                            for ax2_fused_2, ax1 in T.grid(T.int64(2), T.int64(4)):
                                for ax2_fused_0_ax2_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                        with T.block("NT_matmul"):
                                            vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                                            v0 = T.axis.spatial((seq_len + T.int64(3)) // T.int64(4) * T.int64(4), ax0_0 * T.int64(4) + ax1)
                                            v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax2_fused_0_ax2_fused_1_fused * T.int64(2) + ax2_fused_2)
                                            T.reads(NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1])
                                            T.writes(NT_matmul_intermediate_pad_local[T.int64(0), v0, v1])
                                            with T.init():
                                                NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] = T.float16(0)
                                            NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] = NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] + NT_matmul_intermediate_pad_rf_local_1[vax2_fused_u_fused_1_ax2_fused_u_fused_3_fused_0, T.int64(0), v0, v1]
                            for ax0 in range(T.int64(4)):
                                for ax1_fused_0_ax1_fused_1_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax1_fused_2 in range(T.int64(2)):
                                        with T.block("NT_matmul_intermediate_pad"):
                                            v0 = T.axis.spatial(seq_len, ax0_0 * T.int64(4) + ax0)
                                            v1 = T.axis.spatial(T.int64(1024), u_fused_ax1_fused_fused_0 * T.int64(32) + ax1_fused_0_ax1_fused_1_fused * T.int64(2) + ax1_fused_2)
                                            T.where((ax0_0 - (seq_len + T.int64(3)) // T.int64(4) < T.int64(0) or ax0_0 == T.int64(0)) and ax0_0 * T.int64(4) + ax0 < seq_len)
                                            T.reads(NT_matmul_intermediate_pad_local[T.int64(0), v0, v1], transformer_h_0_attn_c_proj_bias3[v1], add290[T.int64(0), v0, v1])
                                            T.writes(T_add_intermediate_1_intermediate[T.int64(0), v0, v1])
                                            T_add_intermediate_1_intermediate[T.int64(0), v0, v1] = NT_matmul_intermediate_pad_local[T.int64(0), v0, v1] + transformer_h_0_attn_c_proj_bias3[v1] + add290[T.int64(0), v0, v1]
            else:
                with T.block("root"):
                    T.reads()
                    T.writes()
                    reshape195_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(1024)), "float16", scope="shared.dyn")
                    dequantize_intermediate_intermediate_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1024)), "float16", scope="shared.dyn")
                    reshape195_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(1024)), "float16", scope="wmma.matrix_a")
                    dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1024)), "float16", scope="wmma.matrix_b")
                    NT_matmul_intermediate_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(1024)), "float16", scope="shared.dyn")
                    NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), (seq_len + T.int64(127)) // T.int64(128) * T.int64(128), T.int64(1024)), "float16", scope="wmma.accumulator")
                    for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
                        for ax1_0_0_ax2_0_0_fused in T.thread_binding((seq_len + T.int64(127)) // T.int64(128), thread="blockIdx.x"):
                            for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(8), thread="blockIdx.y"):
                                for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(2), T.int64(2)):
                                        with T.block("NT_matmul_o_init"):
                                            v0_o = T.axis.spatial(T.int64(1), ax0)
                                            v1_o = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3_init)
                                            v2_o = T.axis.spatial(T.int64(64), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3_init)
                                            T.reads()
                                            T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            with T.block("NT_matmul_init_o"):
                                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                T.reads()
                                                T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                                T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                                    for ax3_0_0 in range(T.int64(16)):
                                        for ax0_ax1_fused_0 in range(T.int64(4)):
                                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                                        with T.block("reshape195_reindex_pad_shared.dyn"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                            v2 = T.axis.spatial(T.int64(1024), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                            T.reads(reshape195[v0, v1, v2])
                                                            T.writes(reshape195_reindex_pad_shared_dyn[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                            reshape195_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v1 < seq_len, reshape195[v0, v1, v2], T.float16(0))
                                        for ax0_ax1_fused_0 in range(T.int64(4)):
                                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                                        with T.block("dequantize_intermediate_intermediate_reindex_shared.dyn"):
                                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                            v1 = T.axis.spatial(T.int64(1024), ax1_0_1_ax2_0_1_fused * T.int64(128) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(64))
                                                            v2 = T.axis.spatial(T.int64(1024), ax3_0_0 * T.int64(64) + (ax0_ax1_fused_0 * T.int64(2048) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(64))
                                                            T.reads(lv397[v1, v2 // T.int64(8)], lv398[v1, v2 // T.int64(32)])
                                                            T.writes(dequantize_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2])
                                                            T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]], "double_buffer_scope": 0, "tir.manifest_shared_memory_local_stage": 1})
                                                            dequantize_intermediate_intermediate_reindex_shared_dyn[v0, v1, v2] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv397[v1, v2 // T.int64(8)], T.Cast("uint32", v2 % T.int64(8) * T.int64(4))), T.uint32(15))) - T.float16(7)) * lv398[v1, v2 // T.int64(32)]
                                        for ax3_0_1 in range(T.int64(4)):
                                            for ax0_0 in T.unroll(T.int64(2)):
                                                for ax1_0 in T.unroll(T.int64(1)):
                                                    with T.block("reshape195_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                                        v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1_o = T.axis.spatial(T.int64(8) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                                        v2_o = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                                        T.reads(reshape195_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        T.writes(reshape195_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        A = T.match_buffer(reshape195_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                                        C = T.match_buffer(reshape195_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                        T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "row_major")
                                            for ax0_0 in T.unroll(T.int64(2)):
                                                for ax1_0 in T.unroll(T.int64(1)):
                                                    with T.block("dequantize_intermediate_intermediate_reindex_shared.dyn_wmma.matrix_b_o"):
                                                        v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v1_o = T.axis.spatial(T.int64(64), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax0_0)
                                                        v2_o = T.axis.spatial(T.int64(64), ax3_0_0 * T.int64(4) + ax3_0_1 + ax1_0)
                                                        T.reads(dequantize_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        T.writes(dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        A = T.match_buffer(dequantize_intermediate_intermediate_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                                        C = T.match_buffer(dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                        T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A.data, A.elem_offset, A.strides[0] * T.int64(16), 1), A.strides[0], "col_major")
                                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(2), T.int64(2)):
                                                with T.block("NT_matmul_o_update"):
                                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                                    v1_o = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(8), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax1_0_3)
                                                    v2_o = T.axis.spatial(T.int64(64), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax2_0_3)
                                                    v3_o = T.axis.reduce(T.int64(64), ax3_0_0 * T.int64(4) + ax3_0_1)
                                                    T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], reshape195_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                    T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                    with T.block("NT_matmul_o"):
                                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                                        T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], reshape195_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                                        T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                                        A = T.match_buffer(reshape195_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                                        B = T.match_buffer(dequantize_intermediate_intermediate_reindex_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                                        C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                                        T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A.data, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), B.data, B.elem_offset // B.strides[0] // T.int64(16) * (B.strides[0] // T.int64(16)) + B.elem_offset % B.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                                        with T.block("NT_matmul_intermediate_reindex_pad_shared.dyn_wmma.accumulator_o"):
                                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1_o = T.axis.spatial(T.int64(8) * ((seq_len + T.int64(127)) // T.int64(128)), ax1_0_0_ax2_0_0_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(2) + ax0_0)
                                            v2_o = T.axis.spatial(T.int64(64), ax1_0_1_ax2_0_1_fused * T.int64(8) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(2) + ax1_0)
                                            T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            T.writes(NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                            A = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                                            C = T.match_buffer(NT_matmul_intermediate_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                                            T.tvm_store_matrix_sync(A.data, 16, 16, 16, A.elem_offset // A.strides[0] // T.int64(16) * (A.strides[0] // T.int64(16)) + A.elem_offset % A.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                                    for ax0_ax1_fused_0 in range(T.int64(8)):
                                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                            for ax0_ax1_fused_2 in T.vectorized(T.int64(4)):
                                                with T.block("NT_matmul_intermediate_reindex_pad_shared.dyn"):
                                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                                    v1 = T.axis.spatial((seq_len + T.int64(127)) // T.int64(128) * T.int64(128), ax1_0_0_ax2_0_0_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused % T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) // T.int64(32))
                                                    v2 = T.axis.spatial(T.int64(1024), ax1_0_1_ax2_0_1_fused * T.int64(128) + ax2_0_2_ax1_0_2_fused // T.int64(4) * T.int64(32) + (ax0_ax1_fused_0 * T.int64(128) + ax0_ax1_fused_1 * T.int64(4) + ax0_ax1_fused_2) % T.int64(32))
                                                    T.reads(NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2], transformer_h_0_attn_c_proj_bias3[v2], add290[T.int64(0), v1, v2])
                                                    T.writes(T_add_intermediate_1_intermediate[T.int64(0), v1, v2])
                                                    T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                                    if v1 < seq_len:
                                                        T_add_intermediate_1_intermediate[T.int64(0), v1, v2] = NT_matmul_intermediate_reindex_pad_shared_dyn[v0, v1, v2] + transformer_h_0_attn_c_proj_bias3[v2] + add290[T.int64(0), v1, v2]
```

This code snippet is from the last phase where the IRModule has been lowered to VM bytecode and ready to be exported as a .so library file for native hardware. What differs this from the one in phase 3 is that it's scheduled for GPU. The code that makes the transformation is the Dlight's scheduling for GPU. It works mainly on the structure of the loops and optimizes them in such a way that they are efficient for the GPU they are going to be run on.

Apart from the loops's structure and additional variables like local or shared, it doesn't really change much. The reason we are stating this is that, if one wants to extract workload dimension, we can do so during the phase 3, by adding some hooks to capture the dimensions. TVM's TIR offers stmt_functor, which permits us to modify the block inside the IRModule's TIR function.

## Modify the compilation process

As mentioned above, to capture the workload dimension, we can do that in phase 3; below is the modified code within the pipeline.py with the added pass, ExtractWorkloadDimension(). 

**Consideration** : 
1. After the TIRs are optimized, the workload dimensions can be easily retrieved, from the computation block with string matmul in it.
2. GPU scheduling use these same dimensions 

```python
    _LogProgress("Lowering to TVM TIR kernels"),
    tvm.relax.backend.DispatchSampling(),
    tvm.relax.backend.DispatchSortScan(),
    tvm.relax.transform.LegalizeOps(),
    tvm.relax.transform.AnnotateTIROpPattern(),
    tvm.relax.transform.FoldConstant(),
    tvm.relax.transform.FuseOps(),
    tvm.relax.transform.FuseTIR(),
    _DebugDump("debug-phase2.py", debug_dump, show_meta=False),
    # Phase 3. Passes on TIR
    _LogProgress("Running TVM TIR-level optimizations"),
    FuseDequantizeMatmulEwise(),
    FuseDequantizeTake(),
    tvm.relax.transform.DeadCodeElimination(),
    CleanUpTIRAttrs(["op_pattern"]),
    ExtractWorkloadDimension(), ## call custom pass to collect workload dimension
    _DebugDump("debug-phase3.py", debug_dump, show_meta=False),
```

### Extracting the workload dimensions
In order to get the workload dimension of LLM model, since their shape is dynamic, we need to do so during the runtime. Hence, adding a hook into the IRModule is necessare; the hook will help us capture the data we need. 

To modify the body of a primitive function, like adding hook, add additional tensor buffer etc. we can use a couple of statement(stmt) of the Abstract syntax tree(AST) node in TVM, they can be found from [stmt.py]([link](https://github.com/mlc-ai/relax/blob/e0ef1c92add4048823a5e2c8724495418865986b/python/tvm/tir/stmt.py)) and the mutator functions from [stmt_functor.py]([link](https://github.com/mlc-ai/relax/blob/e0ef1c92add4048823a5e2c8724495418865986b/python/tvm/tir/stmt_functor.py)) can help us traverse the module/block in pre_order or post_order. 

In my case, the primary goal is to add a hook that will store the workload dimensions, from matrix multiplication, in a file, json or txt. the code is as follow:

* collect_and_insert_call_extern(stmt, func_name) will return the function block that was passed to modify
* pre_order(op) will collect the loopbound of the for loops, store the one of the matmul_block
* post_order(op) only traverse and return the op without modifying the block
* recursive tir.stmt_functor.ir_transform(stmt, pre_order, post_order, None) to traverse the block statement to modify, and return the modified block
* add_preorder(op) to write the stored list containing the dimensions at the beginning of the 
   
```python 
    import tvm
    from tvm import tir
    from tvm.ir.module import IRModule
    from tvm.tir import stmt_functor

    # Define the transformation function to collect dimensions and insert T.call_extern at the top
    def collect_and_insert_call_extern(stmt, func_name):
        loop_extents = []  # Store loop extents
        calls_to_insert = None  # To store T.call_extern statements that will be inserted later
        t_call_already_added = False

        def pre_order(op):
            nonlocal loop_extents, calls_to_insert

            #Store loop content if encounter one
            if isinstance(op, tir.For):
                loop_extents.append(op.extent) 
                return None  # Continue traversal

            # If the block associated with the loop is not matmul, reset the list, else store it to loop data
            if isinstance(op, tir.Block):
                if "matmul" not in op.name_hint.lower():
                    loop_extents = []
                    return None  # Continue traversal
                else:
                    #print(f"Found matmul block: {op.name_hint}")
                    loop_data = []
                    dynamic_args = []
                    for extent in loop_extents:
                        if isinstance(extent, tvm.tir.IntImm):  # Static value
                            loop_data.append(str(extent.value))
                        else:  # Dynamic variable
                            loop_data.append("%d")  # Capture runtime value using variable name
                            dynamic_args.append(extent) #save the dynamic value's name

                    # Construct the log message
                    loop_data_str = f"{func_name}: " + ", ".join(loop_data)
                    #print(f"Logging loop data: {loop_data_str}")

                    # Create the system call to log this information
                    file_path = "/home/jjlab/loop_data.txt"
                    command = f'echo "{loop_data_str}" >> {file_path}'
                    
                    # Log all at once, with dynamic variables inserted inline
                    system_call = tir.Evaluate(tir.call_extern("int32", "system", command, *dynamic_args))
                    # Store the system call to be inserted later at the top of the function
                    calls_to_insert = system_call

                return None  # Do not insert log within matmul block

            return None

        def post_order(op):
            # No post-order operations needed in this case
            return op

        # Step 1: Traverse the function and collect dimensions from matmul blocks
        new_stmt = tir.stmt_functor.ir_transform(stmt, pre_order, post_order, None)


        
        #print(type)
        # Step 2: After traversal, insert the collected T.call_extern statements only at the top of the function
        if calls_to_insert:
            def add_preorder(op):
                #already_added = None
                nonlocal t_call_already_added
                if isinstance(op, tir.For):
                    if not t_call_already_added:
                        t_call_already_added = True
                        return tir.SeqStmt([calls_to_insert, op])
                    else:
                        return op
                return None
            
            return tir.stmt_functor.ir_transform(stmt, add_preorder, None, None)
        #print(type(new_stmt))
        return new_stmt

    # Apply the custom pass to the IRModule
    def apply_custom_pass(mod):
        for gv in mod.functions:
            func = mod[gv]
            if isinstance(func, tir.PrimFunc):
                try:
                    # Apply print statement insertion and call-extern movement
                    func_name = gv.name_hint
                    new_body = collect_and_insert_call_extern(func.body, func_name)
                    mod[gv] = func.with_body(new_body)  # Update the module with the modified function
                except Exception as e:
                    print(f"Error transforming function {gv}: {e}")
        return mod

    # Define the custom pass using TVM's module_pass
    @tvm.ir.transform.module_pass(opt_level=0, name="ExtractWorkloadDimension")
    class ExtractWorkloadDimension:
        def transform_module(self, mod, _ctx: tvm.transform.PassContext) -> IRModule:
            return apply_custom_pass(mod)
```

After modifying pipeline.py we get this after phase-3:

```python
    def fused_fused_dequantize1_fused_NT_matmul6_add7_add8(lv397: T.Buffer((T.int64(1024), T.int64(128)), "uint32"), lv398: T.Buffer((T.int64(1024), T.int64(32)), "float16"), p_reshape195: T.handle, transformer_h_0_attn_c_proj_bias3: T.Buffer((T.int64(1024),), "float16"), p_add290: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        seq_len = T.int64()
        reshape195 = T.match_buffer(p_reshape195, (T.int64(1), seq_len, T.int64(1024)), "float16")
        add290 = T.match_buffer(p_add290, (T.int64(1), seq_len, T.int64(1024)), "float16")
        T_add_intermediate_1_intermediate = T.match_buffer(p_output0, (T.int64(1), seq_len, T.int64(1024)), "float16")
        # with T.block("root"):
        compute = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
        dequantize_intermediate_intermediate = T.alloc_buffer((T.int64(1024), T.int64(1024)), "float16")
        NT_matmul_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
        T_add_intermediate = T.alloc_buffer((T.int64(1), seq_len, T.int64(1024)), "float16")
        T.call_extern("int32", "system", "echo \"fused_fused_dequantize1_fused_NT_matmul6_add7_add8: 1, %d, 1024, 1024\" >> /home/jjlab/loop_data.txt", seq_len)
        for i0, i1 in T.grid(T.int64(1024), T.int64(1024)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv397[v_i0, v_i1 // T.int64(8)])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.Cast("float16", T.bitwise_and(T.shift_right(lv397[v_i0, v_i1 // T.int64(8)], T.Cast("uint32", v_i1 % T.int64(8) * T.int64(4))), T.uint32(15)))
        for i0, i1 in T.grid(T.int64(1024), T.int64(1024)):
            with T.block("dequantize"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(compute[v_i0, v_i1], lv398[v_i0, v_i1 // T.int64(32)])
                T.writes(dequantize_intermediate_intermediate[v_i0, v_i1])
                dequantize_intermediate_intermediate[v_i0, v_i1] = (compute[v_i0, v_i1] - T.float16(7)) * lv398[v_i0, v_i1 // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), seq_len, T.int64(1024), T.int64(1024)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(reshape195[v_i0, v_i1, v_k], dequantize_intermediate_intermediate[v_i2, v_k])
                T.writes(NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                NT_matmul_intermediate[v_i0, v_i1, v_i2] = NT_matmul_intermediate[v_i0, v_i1, v_i2] + reshape195[v_i0, v_i1, v_k] * dequantize_intermediate_intermediate[v_i2, v_k]
```

### To do:
1. Run a model on relax VM to test the extraction
2. Study quantization techniques