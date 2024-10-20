import minitorch
from minitorch import tensor
from functools import partial

if __name__ == '__main__':
    FastTensorBackend = minitorch.TensorBackend(minitorch.CudaOps)
    cuTensor = partial(tensor, backend=FastTensorBackend)
    a = cuTensor([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ])
    b = cuTensor([
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])

    print(a @ b)
    # print(b @ a)