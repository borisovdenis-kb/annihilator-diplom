from src.utility.notify_execution_time import get_execution_time
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
    mat_list = MathUtility.get_matrix_with_test_data_cpu(BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH)
    execution_time = get_execution_time(get_zhegalkin_cpu(mat_list, BOOL_VECTOR_LENGTH))
    print(execution_time)
