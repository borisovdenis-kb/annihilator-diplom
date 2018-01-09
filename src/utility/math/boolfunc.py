from abc import ABCMeta, abstractmethod
from src.utility.math import math_utils
import math


class BoolFunction:
    __metaclass__ = ABCMeta

    def __init__(self, var_list):
        """
        :param var_list - list of variables
        example - f(x, y, z); var_list = [x, y, z]
        """
        self.var_list = var_list
        self.var_amount = len(self.__var_list)

    @abstractmethod
    def __mul__(self, other):
        """Multiplication of boolean functions"""

    @abstractmethod
    def __xor__(self, other):
        """Addiction of boolean function in GF(2)"""

    @property
    def var_list(self):
        return self.__var_list

    @var_list.setter
    def var_list(self, var_list):
        self.__var_list = var_list

    @property
    def var_amount(self):
        return self.__var_amount

    @var_amount.setter
    def var_amount(self, var_amount):
        self.__var_amount = var_amount


class VectorBoolFunction(BoolFunction):
    def __init__(self, var_list, values_vector, anf=list()):
        """
        :param var_list - list of variables
        example - f(x, y, z); var_list = [x, y, z]
        :param vector: list of bits 
        example: [0, 1, 1, 0] vector representation of boolean function x + y (ANF)
        """
        super().__init__(var_list)
        self.values_vector = values_vector

        if anf:
            self.anf = anf
        else:
            self.anf = math_utils.find_zhegalkin_poly_coefficients(values_vector)

    def __xor__(self, other):
        """
        Multiplication of two polynomials in bit vectors form.
        :param other: list of bit
        example: [0, 0, 0, 1] vector representation of boolean function x & y
        :return: list of bit - [0, 0, 0, 0]
        """
        res = [0] * len(self.anf)
        for i in range(len(self.anf)):
            res[i] = self.anf[i] ^ other[i]
        return VectorBoolFunction(self.var_list, res)

    def __mul__(self, other):
        """
        Multiplication of two polynomials in bit vectors form.
        :param other: list of bit
        example: [0, 0, 0, 1] vector representation of boolean function x & y
        :return: list of bit - [0, 0, 0, 0]
        """
        res = [0] * len(self.anf)
        for i in range(len(self.anf)):
            res[i] = self.anf[i] & other[i]
        return VectorBoolFunction(self.var_list, res)

    def __getitem__(self, idx):
        if idx > len(self.__vector) or idx < 0:
            raise IndexError
        return self.__vector[idx]

    def __str__(self):
        return self.anf.__str__()

    def __repr__(self):
        return self.anf.__repr__()

    def scalar_product(self, other_func):
        vector_a = self.anf
        vector_b = other_func.anf

        result = 0
        for x, y in zip(vector_a, vector_b):
            result ^= x & y

        return result

    # TODO: add function for getting function's values

    @staticmethod
    def check_vector_lenght(vector, var_amount):
        if len(vector) != 2**var_amount:
            raise ValueError(
                "len of boolean vector of f(x1, ..., xN) must be 2**N"
            )

    @property
    def anf(self):
        return self.__vector

    @anf.setter
    def anf(self, vector):
        VectorBoolFunction.check_vector_lenght(vector, self.var_amount)
        self.__vector = vector


class SymbolicBoolFunction(BoolFunction):
    def __init__(self, var_list, func):
        """
        :param var_list - list of variables
        example - f(x, y, z); var_list = [x, y, z]
        :param func: symbolic representation of boolean function
        example - x + y; x*y + y (ANF)
        """
        super().__init__(var_list)
        self.func = func

    def __xor__(self, other):
        return SymbolicBoolFunction(
            self.var_list,
            math_utils.add_polynomials(self.func, other.func, self.var_list)
        )

    def __mul__(self, other):
        return SymbolicBoolFunction(
            self.var_list,
            math_utils.multiply_polynomials(self.func, other.func, self.var_list)
        )

    def __str__(self):
        return self.func

    def __repr__(self):
        return self.func

    @property
    def func(self):
        return self.__func

    @func.setter
    def func(self, func):
        self.__func = func


class BoolFuncConverter:
    @staticmethod
    def convert_vector_to_symbolic(func):
        """
        :param func: object of type VectorBoolFunction
        :return: anf of boolean function
        example: [0, 1, 1, 0] - x + y
        """
        zheg_poly_coefs = math_utils.find_zhegalkin_poly_coefficients(func.anf)
        n = func.var_amount
        return math_utils.build_zhegalkin_poly(func.anf, func.var_list, generate_bool_vectors_from_numbers(n, 2**n))


def find_function_non_linearity(func):
    """
    :param func: object of type VectorBoolFunction
    :return: int
    """
    coefficients = find_walsh_hadamard_coefficients(func)
    n = func.var_amount
    return 2**(n-1) - (max(*coefficients) // 2)


def find_walsh_hadamard_coefficients(func):
    """
    :param func: object of type VectorBoolFunction
    :return: list of int
    example: func - [0, 1, 1, 1]
    """
    func_values = func.values_vector
    n = func.var_amount
    vectors = generate_bool_vectors_from_numbers(n, 2**n)

    coefficients = []
    for v in vectors:
        SUM = 0
        for i, u in enumerate(vectors):
            SUM += (-1)**(get_scalar_product(v, u) ^ func_values[i])
        coefficients.append(abs(SUM))
    return coefficients


def find_function_degree(func):
    """
    :param func: symbolic
    :return: 
    """
    string = func.__repr__() + "+"

    degree, count = 1, 1
    for c in string:
        if c == "*":
            count += 1
        elif c == "+":
            if count > degree:
                degree = count
            count = 0
    return degree


def generate_bool_vectors_from_numbers(n, size):
    """
    :param n: amount of variables of boolean function
    :return: 
    """
    result = []
    N = 2**n
    for i in range(N):
        s = bin(i)[2:]
        bits = '0' * (size - len(s)) + '%s' % s
        result.append([int(x) for x in list(bits)])
    return result


def get_scalar_product(vector1, vector2):
    result = 0
    for x, y in zip(vector1, vector2):
        result ^= x & y
    return result


def test_equation_system_generator():
    print("Boolean function in vector representation:")
    bool_vector_one = VectorBoolFunction(['x', 'y'], [0, 1, 1, 0])
    bool_vector_two = VectorBoolFunction(['x', 'y'], [0, 0, 1, 0])
    print(bool_vector_one, " * ", bool_vector_two)
    print("mul:", bool_vector_one * bool_vector_two)
    print(bool_vector_one, " + ", bool_vector_two)
    print("add:", bool_vector_one ^ bool_vector_two)

    print()

    print("Boolean function in symbolic representation:")
    bool_func_one = SymbolicBoolFunction(['x', 'y'], "x + x*y")
    bool_func_two = SymbolicBoolFunction(['x', 'y'], "0")
    print(bool_func_one, " * ", bool_func_two)
    print("mul:", bool_func_one * bool_func_two)
    print(bool_func_one, " + ", bool_func_two)
    print("add:", bool_func_one ^ bool_func_two)


def test_generate_bool_vectors_from_numbers(n, size):
    for vector in generate_bool_vectors_from_numbers(n, size):
        print(vector)


def test_boll_func_converter():
    v = [0, 1, 1, 0]
    print(BoolFuncConverter.convert_vector_to_symbolic(v, 2))


def test_find_walsh_hadamard_coefficients():
    func = VectorBoolFunction(['x', 'y', 'z'], [0, 1, 1, 0, 1, 1, 0, 1])
    print(find_walsh_hadamard_coefficients(func))


def test_find_function_non_linearity():
    func = VectorBoolFunction(['x', 'y', 'z'], [1, 0, 0, 1, 0, 0, 1, 1])
    print(find_function_non_linearity(func))


if __name__ == "__main__":
    # test_equation_system_generator()
    # test_boll_func_converter()
    # test_generate_bool_vectors_from_numbers(3, 8)
    # test_find_walsh_hadamard_coefficients()
    test_find_function_non_linearity()
