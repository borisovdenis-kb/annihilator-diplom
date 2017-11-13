import pycuda.driver as drv
from time import time
from pycuda.compiler import SourceModule
from src.utility.external_func import *

L = 32
N = 10000

mod = SourceModule("""

__global__ void find_null_vectors(long* null_vectors, vector, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    if (i < N) {
        if (i & vector) {
            null_vectors[count] = i;
            count += 1;
        }
    }
}

""")


if __name__ == '__main__':
    null_vectors = numpy.array()

    find_null_vectors = mod.get_function("find_null_vectors")

    start = time()
    find_null_vectors(
        drv.Out(input_matrix),
        block=(1024, 1, 1), grid=(2056, 1)
    )
    end = time()

    print(end - start)
