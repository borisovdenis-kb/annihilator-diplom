import numpy
from src.constants import BOOL_VECTOR_LENGTH, BOOL_VECTOR_AMOUNT


class MathUtility:
    @staticmethod
    def func_vectors_to_matrix(vectors):
        pass

    @staticmethod
    def zheg_rdy2mul(bool_vector, l):
        mat = numpy.zeros(shape=(l, l))
        for i in range(l):
            mat[i][0] = bool_vector[i]

        for i in range(l):
            for j in range(l):
                mat[i][j] = mat[i][0]

        return mat

    @staticmethod
    def null_rdy2mul(vec, l):
        mat = numpy.zeros(shape=(l, l))
        for i in range(l):
            mat[0][i] = vec[i]

        for i in range(l):
            for j in range(l):
                mat[i][j] = mat[0][j]

        return mat

    @staticmethod
    def matrix_to_zheg_vectors(mat, n, l):
        """
            Функция на вход принимает результат нахождения 
            многочлена Жегалкина(матрицу).
            Функция берет из первого столбца матрицы по L значений
            и записывает их в вектора.
            :param mat: 
            :param n: 
            :param l: 
            :return: 
        """
        vec = []
        vectors = []
        for i in range(n*l):
            if i % l == 0 and i != 0:
                vectors.append(vec)
                vec = []
            vec.append(mat[i][0])

        return numpy.array(vectors)

    @staticmethod
    def get_matrix_with_test_data_gpu(vectors_amount, vector_length, outfile="resources/input/func_in_gpu.txt"):
        """
            Функция заполнения матриц для gpu.
            Матрица имеет вид:
            10100...1
            0.......0
            0.......0
            0.......0
            .........
            000000000
            Где первой строкой явлется вектор некотороый 
            булевой функции.
            :param vectors_amount: количество векторов
            :param vector_length: длина вектора
            :param outfile: 
        """
        big_matrix = numpy.random.randint(1, size=(vectors_amount * vector_length, vector_length), dtype=int)

        for i in range(0, vectors_amount * vector_length, vector_length):
            big_matrix[i][0] = 1
            big_matrix[i][2] = 1
            big_matrix[i][4] = 1
            big_matrix[i][7] = 1

        numpy.savetxt(outfile, big_matrix, fmt='%1.0f')

        return big_matrix

    @staticmethod
    def get_matrix_with_test_data_cpu(vectors_amount, vector_length):
        """
            Функция заполнения матриц для сpu.
            :param vectors_amount: количество векторов
            :param vector_length: длина вектора
        """
        list_of_matrix = []

        for i in range(vectors_amount):
            mat = numpy.random.randint(1, size=(vector_length, vector_length), dtype=int)

            for j in range(vector_length):
                mat[j][0] = 1
                mat[j][2] = 1
                mat[j][4] = 1
                mat[j][7] = 1

            list_of_matrix.append(mat)

        return list_of_matrix

    @staticmethod
    def dim(vector):
        return len(vector)

    @staticmethod
    def find_all_nulling_vectors(vector):
        # для заданного вектора v находит все возможные векторы z
        # такие что, если v[i] = 1, z[i] = 0; если v[i] = 0, z[i] = 0 or 1
        # кроме нулевого
        # res = [[1,0,1,0], ...] - список списков.
        n = MathUtility.dim(vector)
        vector_in_int = int(''.join([str(x) for x in vector]), 2)
        res = []

        for i in range(2**n):
            if vector_in_int & i == 0:
                nulling_vector = list('0'*(n-len(bin(i)[2:])) + bin(i)[2:])
                res.append(list(map(int, nulling_vector)))

        return res[1:]


if __name__ == '__main__':
    MathUtility.get_matrix_with_test_data_gpu(BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH)
    # print(numpy.loadtxt('func_in_input.txt'))

    mat_list = MathUtility.get_matrix_with_test_data_cpu(BOOL_VECTOR_AMOUNT, BOOL_VECTOR_LENGTH)
    print(mat_list)
