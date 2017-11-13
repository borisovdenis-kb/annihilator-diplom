import random
import numpy
import pycuda.driver as drv
from time import time
from pycuda.compiler import SourceModule
from src.constants import BOOL_VECTOR_LENGTH, BOOL_VECTOR_AMOUNT
from src.utility.external_func import MathUtility


mod = SourceModule("""
#define vectors 100
#define L 8
#define N L*vectors


__global__ void get_zheg_poly(long A[N][L])
{
    int b = threadIdx.x + blockIdx.x * blockDim.x;
    if (b < vectors) {
        int count = L * b + 1;                      
        for (int i = count; i < count + L - 1; i++) {
            for (int j = 0; j < L - 1; j++) {
                A[i][j] = A[i - 1][j] ^ A[i - 1][j + 1];
            }
        }
    }
}

__global__ void mul_zheg_poly(int A[L][L], int B[L][L], int C[L][L])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i < L && j < L)
        C[i][j] = A[i][j] * B[i][j];
}


_global__ void find_null_vectors(long* null_vectors, vector, int N)
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
    input_matrix = MathUtility.get_matrix_with_test_data_gpu(BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH).astype(numpy.float32)

    get_zheg_poly = mod.get_function("get_zheg_poly")

    start = time()
    get_zheg_poly(
        drv.InOut(input_matrix),
        block=(1024, 1, 1), grid=(2056, 1)
    )
    end = time()
    print('run_time get_zheg_poly [sec]: ', end - start)

    zheg_vectors = MathUtility.matrix_to_zheg_vectors(input_matrix, BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH)

    numpy.savetxt("resources/output/test_result.txt", input_matrix, fmt='%1.0f')
    numpy.savetxt("resources/output/zheg_vectors.txt", zheg_vectors, fmt='%s')

    zheg_vectors = numpy.loadtxt("txt_files/zheg_vectors.txt").astype(numpy.float32)

    # TODO переделать, использовать find_null_vectors
    # from
    test_null_vector = []
    for i in range(BOOL_VECTOR_LENGTH):
        test_null_vector.append(random.randint(0, 1))

    print(test_null_vector)
    res_time = 0
    # to

    for i in range(len(zheg_vectors)):
        bool_vector_a = MathUtility.zheg_rdy2mul(zheg_vectors[i], BOOL_VECTOR_LENGTH)
        bool_vector_b = MathUtility.null_rdy2mul(test_null_vector, BOOL_VECTOR_LENGTH)
        result = numpy.zeros_like(bool_vector_a, dtype=numpy.int)

        mul_zheg = mod.get_function("mul_zheg_poly")
        start_mul1 = time()
        mul_zheg(
            drv.In(bool_vector_a.astype(numpy.int)), drv.In(bool_vector_b.astype(numpy.int)), drv.InOut(result),
            block=(32, 32, 1), grid=(32, 32)
        )
        end_mul1 = time()
        res_time += end_mul1 - start_mul1

    print('run_time mul_zheg_poly[sec]: ', res_time)

    # TODO Решение СНЛУ

    print('Finish')
