from time import time
from src.utility.external_func import MathUtility
from src.constants import BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH


def get_zhegalkin_cpu(mat_list, l):
    res = []
    for mat in mat_list:
        for i in range(1, l):
            for j in range(l-i):
                mat[i][j] = mat[i-1][j] ^ mat[i-1][j+1]
        res.append(mat)

    return res


if __name__ == '__main__':
    mat_list = MathUtility.fill_input_matrix_tst_cpu(BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH)

    # TODO: Сделать декоратор для замера времени исполнения.
    start = time()
    res = get_zhegalkin_cpu(mat_list, BOOL_VECTOR_LENGTH)
    end = time()

    print(end - start)
