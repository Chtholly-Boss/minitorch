import minitorch
from minitorch import tensor

if __name__ == '__main__':
    FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
    # a = tensor([
    #     [1, 2],
    #     [1, 1]
    # ], backend=FastTensorBackend)
    a = tensor([
        [
            [1, 0, 1],
            [0, 1, 1]
        ],
        [
            [2, 0, 2],
            [0, 2, 2]
        ]
    ], backend=FastTensorBackend)
    b = tensor([
        [2, 2],
        [4, 4],
    ], backend=FastTensorBackend)
    c = b @ a
    print(c)