import minitorch
from minitorch import tensor

if __name__ == '__main__':
    FastTensorBackend = minitorch.TensorBackend(minitorch.CudaOps)
    a = tensor([
        [1, 2],
        [1, 1]
    ], backend=FastTensorBackend)
    b = a.f.neg_map(a)
    print(b)