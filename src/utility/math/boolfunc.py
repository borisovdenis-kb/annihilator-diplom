# from src.utility.math import math_utils
from abc import ABCMeta, abstractmethod
from src.utility.math import math_utils


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
    def __init__(self, var_list, vector):
        """
        :param var_list - list of variables
        example - f(x, y, z); var_list = [x, y, z]
        :param vector: list of bits 
        example: [0, 1, 1, 0] vector representation of boolean function x + y
        """
        super().__init__(var_list)
        self.vector = vector

    def __xor__(self, other):
        """
        Multiplication of two polynomials in bit vectors form.
        :param other: list of bit
        example: [0, 0, 0, 1] vector representation of boolean function x & y
        :return: list of bit - [0, 0, 0, 0]
        """
        res = [0] * len(self.vector)
        for i in range(len(self.vector)):
            res[i] = self.vector[i] ^ other[i]
        return VectorBoolFunction(self.var_list, res)

    def __mul__(self, other):
        """
        Multiplication of two polynomials in bit vectors form.
        :param other: list of bit
        example: [0, 0, 0, 1] vector representation of boolean function x & y
        :return: list of bit - [0, 0, 0, 0]
        """
        res = [0] * len(self.vector)
        for i in range(len(self.vector)):
            res[i] = self.vector[i] & other[i]
        return VectorBoolFunction(self.var_list, res)

    def __getitem__(self, idx):
        if idx > len(self.__vector) or idx < 0:
            raise IndexError
        return self.__vector[idx]

    def __str__(self):
        return self.vector.__str__()

    def __repr__(self):
        return self.vector.__repr__()

    @property
    def vector(self):
        return self.__vector

    @vector.setter
    def vector(self, vector):
        if len(vector) != 2**self.var_amount:
            raise ValueError(
                "len of boolean vector of f(x1, ..., xN) must be 2**N"
            )
        self.__vector = vector


class SymbolicBoolFunction(BoolFunction):
    def __init__(self, var_list, func):
        """
        :param var_list - list of variables
        example - f(x, y, z); var_list = [x, y, z]
        :param func: symbolic representation of boolean function
        example - x + y; x*y + y
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


if __name__ == "__main__":
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
