# Extract the workload dimensions and tiling factor

From MLC Course we copy this simple matrix multiplication IRModule

```python
    @tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),
             B: T.Buffer((1024, 1024), "float32"),
             C: T.Buffer((1024, 1024), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

In the module, we are multiplying matrices A and B, with size [1024, 1024]. below is the script to schedule the matmul for GPU:

```python
def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch
```

We pass as parameters the tiling block, the tiling factor for block and local for both direction x and y, as well as the tiling factor for k.

The first thing is to target the block we want to modify. In our case the block we want to modify is the C block. the line "C_local" is to increase memory reuse by loading a strip of A and B and store the compute value to C_local

```python
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")
```

The loops are split, reordered and then binded to block and threads.  The number of thread blocks and threads depends on the matrices sizes and the tiling factors. For example in the example above we pass [8, 8, 8, 8, 4] as tiling factors so we'd get the following expressions:

* For i:
  * $i_0$: Ranges from $0$ to $\left\lfloor \frac{I}{(tile\_block\_y \times tile\_local\_y)} \right\rfloor - 1$.
  * $i_1$: Ranges from $0$ to $tile\_block\_y - 1$.
  * $i_2$: Ranges from $0$ to $tile\_locay\_y - 1$
  * Relationship between them is: $i = i_0 \times i_1 \times i_2$

* For j:
  * $j_0$: Ranges from $0$ to $\left\lfloor \frac{J}{(tile\_block\_x \times tile\_local\_x)} \right\rfloor - 1$.
  * $j_1$: Ranges from $0$ to $tile\_block\_x - 1$.
  * $j_2$: Ranges from $0$ to $tile\_locay\_x - 1$
  * Relationship between them is: $j = j_0 \times j_1 \times j_2$

* For k:
  * $k_0$: Ranges from $0$ to $\frac{K}{tile\_k} - 1$
  * $k_0$: Ranges from $0$ to $tile\_k$
  * $k = k_0 \times k_1$

The block and thread bindings are made so that $i_0$ represents the numbers of blocks along the y-axis and $i_1$, the number of threads along the y-axis. For j, it's the same logic along the x-axis. The inner loops $i_2$ and $j_2$ will be representing the core/grid inside one thread.  

The module after scheduling can be shown below:

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1 in T.thread_binding(8, thread="threadIdx.y"):
                    for j_1 in T.thread_binding(8, thread="threadIdx.x"):
                        for i_2_init, j_2_init in T.grid(8, 8):
                            with T.block("C_init"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2_init)
                                vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2_init)
                                T.reads()
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = T.float32(0)
                        for k_0 in range(256):
                            for k_1 in T.unroll(4):
                                for i_2, j_2 in T.grid(8, 8):
                                    with T.block("C_update"):
                                        vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                                        vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                                        vk = T.axis.reduce(1024, k_0 * 4 + k_1)
                                        T.reads(C_local[vi, vj], A[vi, vk], B[vk, vj])
                                        T.writes(C_local[vi, vj])
                                        C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                        for ax0, ax1 in T.grid(8, 8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
                                v1 = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + ax1)
                                T.reads(C_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_local[v0, v1]
```

and below code shows  the generated cuda code, the host code is missing(don't know how to get it yet) for inspection: the generated code shows four $8 \times 8$ grids, which means that the reduced dimension was unrolled by $k_1 = 4$ while $k_0$ will iterate from $0$ to $256$. 

```c

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(64) main_kernel(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  for (int i_2_init = 0; i_2_init < 8; ++i_2_init) {
    for (int j_2_init = 0; j_2_init < 8; ++j_2_init) {
      C_local[((i_2_init * 8) + j_2_init)] = 0.000000e+00f;
    }
  }
  for (int k_0 = 0; k_0 < 256; ++k_0) {
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      for (int j_2 = 0; j_2 < 8; ++j_2) {
        C_local[((i_2 * 8) + j_2)] = (C_local[((i_2 * 8) + j_2)] + (A[((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_2 * 1024)) + (k_0 * 4))] * B[((((k_0 * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_2)]));
      }
    }
    for (int i_2_1 = 0; i_2_1 < 8; ++i_2_1) {
      for (int j_2_1 = 0; j_2_1 < 8; ++j_2_1) {
        C_local[((i_2_1 * 8) + j_2_1)] = (C_local[((i_2_1 * 8) + j_2_1)] + (A[(((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_2_1 * 1024)) + (k_0 * 4)) + 1)] * B[(((((k_0 * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_2_1) + 1024)]));
      }
    }
    for (int i_2_2 = 0; i_2_2 < 8; ++i_2_2) {
      for (int j_2_2 = 0; j_2_2 < 8; ++j_2_2) {
        C_local[((i_2_2 * 8) + j_2_2)] = (C_local[((i_2_2 * 8) + j_2_2)] + (A[(((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_2_2 * 1024)) + (k_0 * 4)) + 2)] * B[(((((k_0 * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_2_2) + 2048)]));
      }
    }
    for (int i_2_3 = 0; i_2_3 < 8; ++i_2_3) {
      for (int j_2_3 = 0; j_2_3 < 8; ++j_2_3) {
        C_local[((i_2_3 * 8) + j_2_3)] = (C_local[((i_2_3 * 8) + j_2_3)] + (A[(((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (i_2_3 * 1024)) + (k_0 * 4)) + 3)] * B[(((((k_0 * 4096) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + j_2_3) + 3072)]));
      }
    }
  }
  for (int ax0 = 0; ax0 < 8; ++ax0) {
    for (int ax1 = 0; ax1 < 8; ++ax1) {
      C[((((((((int)blockIdx.y) * 65536) + (((int)threadIdx.y) * 8192)) + (ax0 * 1024)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) * 8)) + ax1)] = C_local[((ax0 * 8) + ax1)];
    }
  }
}
```

Analogically, consider that the 2D grid is mapped to Hardware primitive. What is needed is to shrink the matrices so that they can fit onto that core. 




