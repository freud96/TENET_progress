import IPython
import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
                 
import ctypes
from ctypes import *
import numpy as np
# Load the shared library
lib = ctypes.CDLL("/home/jjlab/tvm-unity/build/libtvm.so")

# Check if the function is available
if not hasattr(lib, "log_loop_indices"):
    raise RuntimeError("log_loop_indices not found in shared library!")
import torch
import torchvision

test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print("Class:", class_names[label[0]])


def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2

import pickle as pkl

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
res = numpy_mlp(img.reshape(1, 784),
                mlp_params["w0"],
                mlp_params["b0"],
                mlp_params["w1"],
                mlp_params["b1"])
print(res)
pred_kind = res.argmax(axis=1)
print(pred_kind)
print("NumPy-MLP Prediction:", class_names[pred_kind[0]])


def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 128), dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(128):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
     for i in range(1):
        for j in range(128):
            Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 10), dtype="float32")
    for i in range(1):
        for j in range(10):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(10):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out

result =lnumpy_mlp(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = result.argmax(axis=1)
print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])



@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def relu0(x: T.handle, y: T.handle):
        n = T.int64()
        X = T.match_buffer(x, (1, n), "float32")
        Y = T.match_buffer(y, (1, n), "float32")
        for i, j in T.grid(1, n):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(x: T.handle,
                w: T.handle,
                b: T.handle,
                z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        
        #T.evaluate(T.call_extern(
            #"void",  # Return type
            #"log_loop_indices",  # External logging function
            #"linear0", 1, m, n, 4, 4, 4  # Function name and dimensions
        #))
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
def lnumpy_call_dps_packed(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out

def lnumpy_mlp_with_call_dps_packed(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_dps_packed(lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_dps_packed(lnumpy_relu0, (lv0, ), (1, 128), dtype="float32")
    out = lnumpy_call_dps_packed(lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out

result = lnumpy_mlp_with_call_dps_packed(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = np.argmax(result, axis=1)
print("Low-level Numpy with CallTIR Prediction:", class_names[pred_kind[0]])


mod = MyModule
#.build(MyInstrumentedModule, target="llvm")

mod.show()

print(mod.script(show_meta=True))
#Prepare input data
#A = tvm.nd.array(np.random.rand(4, 4).astype("float32"))
#B = tvm.nd.array(np.random.rand(4, 4).astype("float32"))
#C = tvm.nd.array(np.zeros((4, 4), dtype="float32"))

target = tvm.target.Target("llvm", host="llvm")
ex = relax.build(mod, target=target)
print("Executable Statistics:", ex.stats())
print(ex.as_text())
## Execute the VM
#mod(A, B, C)

vm = relax.VirtualMachine(ex, tvm.cpu(), profile=True)
data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

func_name = "main"
vm.set_input(func_name, data_nd, nd_params["w0"], nd_params["b0"], 
             nd_params["w1"], nd_params["b1"])

vm.invoke_stateful(func_name)
output = vm.get_outputs(func_name)
print("Model output")
pred_kind = np.argmax(output.numpy(), axis=1)
print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])
# === Time Evaluator for Overall Performance ===
#evaluator = vm.time_evaluator("invoke_stateful", tvm.cpu())(func_name)
## result = evaluator(data_nd, 
##                    nd_params["w0"], nd_params["b0"], 
##                    nd_params["w1"], nd_params["b1"])

#print(f"time evaluator {evaluator}")
# # Print the overall execution time statistics.
# print(f"Mean execution time: {result.mean * 1e3:.2f} ms")
# print(f"Standard deviation: {result.std * 1e3:.2f} ms")

# === Profiling for Per-Operation Analysis ===
report = vm.profile(func_name)

# Print the per-operation profiling report.
print(report)

## === Make Predictions Based on Model Output ===
#nd_res = vm["main"](data_nd, 
                    #nd_params["w0"], nd_params["b0"], 
                    #nd_params["w1"], nd_params["b1"])

## Get the predicted class.
#pred_kind = np.argmax(nd_res.numpy(), axis=1)
#print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])


## Define the LoopData structure in Python
class LoopData(ctypes.Structure):
    _fields_ = [
        ("func_name", ctypes.c_char * 256),
        ("i", ctypes.c_int),
        ("j", ctypes.c_int),
        ("k", ctypes.c_int),
        ("tile_i", ctypes.c_int),
        ("tile_j", ctypes.c_int),
        ("tile_k", ctypes.c_int),
    ]

# Define get_logged_data to return a pointer to the data
# Define return and argument types for get_logged_data
lib.get_logged_data.restype = ctypes.POINTER(LoopData)
lib.get_logged_data.argtypes = [ctypes.POINTER(ctypes.c_size_t)]

# Define the size variable
size = ctypes.c_size_t()

# Retrieve the logged data
data_ptr = lib.get_logged_data(ctypes.byref(size))


if data_ptr:
    print(f"Retrieved {size.value} elements:")
    # Loop through the retrieved elements and print them
    for i in range(size.value):
        entry = data_ptr[i]
        func_name = entry.func_name.decode('utf-8').rstrip('\x00')  # Clean the name
        print(f"Function: {func_name}, i: {entry.i}, j: {entry.j}, k: {entry.k}, "
              f"tile_i: {entry.tile_i}, tile_j: {entry.tile_j}, tile_k: {entry.tile_k}")

    # Clear the log after processing
    lib.clear_log()
else:
    print("No logs available.")
