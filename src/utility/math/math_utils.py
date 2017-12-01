import sympy
import re
from multipledispatch import dispatch
from src.ciphers import ciphera5
from src.utility.math import boolfunc


class EquationSystemGenerator:

    ALPHABET = "xyztqplvudhabc"

    def __init__(self, *lfsr_list, combine_func=None):
        """
        :param lfsr_list: list of LFSR objects
        :param combine_func: combine function of all lfsr outputs
        example: r1*r2 + r2
        """
        self.lfsr_list = lfsr_list
        self.variables = self.__generate_variables()
        self.combine_func = combine_func

    def generate_equation_system(self, iteration_amount):
        self.__generate_variables()
        self.__init_lfsr_with_variables()

        n = len(self.lfsr_list)
        system = [[] for i in range(n)]

        for i in range(iteration_amount):
            for j in range(n):
                system[j].append(self.lfsr_list[j].func_left_shift())
            # execute combine function
        return system

    def __init_lfsr_with_variables(self):
        for i in range(len(self.lfsr_list)):
            self.lfsr_list[i].lfsr = self.variables[i]

    def __generate_variables(self):
        res_var_lists = []
        for i in range(len(self.lfsr_list)):
            tmp_var_list = []
            for j in range(self.lfsr_list[i].length):
                tmp_var_list.append(EquationSystemGenerator.ALPHABET[i] + str(j))

            res_var_lists.append([])
            for j in range(len(tmp_var_list)):
                res_var_lists[i].append(boolfunc.SymbolicBoolFunction(tmp_var_list, tmp_var_list[j]))
        return res_var_lists


def set_truth_table():
    # функция задания таблицы истинности
    # xy
    # 00 | 0
    # 01 | 1
    # 10 | 1
    # 11 | 0
    # возвращает :
    # res - [переменные, значения переменных, вектор значений функции]
    f = []
    var_values = []
    var = list(input("Enter variables : "))
    n = len(var)
    for i in range(2 ** n):
        s = bin(i)[2:]
        line = '0' * (n - len(s)) + '%s' % s
        f.append(int(input(line + ' | ')))
        var_values.append(list(map(int, list(line))))

    return [var, var_values, f]


def build_zhegalkin_polynomial(vector, VARS, var_values):
    res = []
    coef = find_zhegalkin_polynomial_coefficients(vector)
    if coef[0] == 1:
        res = [str(coef[0])]
    for i, values in enumerate(var_values):
        s = []
        if coef[i] != 0:
            for j in range(len(values)):
                if values[j] == 1:
                    s.append(VARS[j])
            res.append('*'.join(s))

    return '+'.join([s for s in res if s])


def find_zhegalkin_polynomial_coefficients(f):
    # нахождения коэффициентов многочлена Жег.
    # методом Паскаля
    # на вход принимает вектор значений булевой функции f
    # ны выходе вектор коэффициентов многочлена Жег.
    n = len(f)
    coef = [[f[i]] for i in range(n)]

    while n >= 1:
        temp = [[] for i in range(n // 2)]

        if len(coef) == 1:
            return sum(coef, [])

        for i in range(len(coef)):
            if i % 2 == 0:
                for j in range(len(coef[i])):
                    temp[int(i / 2)].append(coef[i][j])
            else:
                for j in range(len(coef[i])):
                    temp[int(i / 2)].append(coef[i - 1][j] ^ coef[i][j])
        n //= 2
        coef = temp


def dim(vector):
    return len(vector)


def show_annig(func, annig):
    # функция демонстрирует работу аннигилятора для функции func
    print()
    print('Annig : %s' % annig[0])
    print('(%s)*(%s) = %s = %s = %s' % (func, annig[0], annig[1], annig[2], annig[3]))


@dispatch(list, list)
def multiply_polynomials(vector1, vector2):
    """
    Multiplication of two polynomials in bit vectors form.
    :param vector1: list of bits - [0, 1, 1, 0] or x + y
    :param vector2: list of bit - [0, 0, 0, 1] or x & y
    :return: list of bit - [0, 0, 0, 0]
    """
    res = [0] * len(vector1)
    for i in range(len(vector1)):
        res[i] = vector1[i] & vector2[i]
    return res


@dispatch(str, str, list)
def multiply_polynomials(p1, p2, var_set):
    """
    :param p1: str - polynomial ["x*y"]
    :param p2: str - polynomial ["x*y + x"]
    :param var_set: list ["x", "y"]
    :return: simplified result of multiplication of p1 and p2
    """
    poly_multiplication = '(' + p1 + ')*(' + p2 + ')'
    res = simplify_poly(poly_multiplication, var_set)
    return res


def add_polynomials(p1, p2, var_set):
    """
    :param p1: str - polynomial ["x*y"]
    :param p2: str - polynomial ["x*y + x"]
    :param var_set: list ["x", "y"]
    :return: simplified result of addiction of p1 and p2
    """
    poly_addiction = p1 + " + " + p2
    res = simplify_poly(poly_addiction, var_set)
    return res


def execute(poly):
    res = sympy.expand(poly)
    return res.__repr__()


# noinspection PyTypeChecker
def simplify_poly(poly, var_set):
    res = execute(poly)

    simplification_steps = [res]

    for v in var_set:
        res = re.sub(r'%s\*\*\d+' % v, v, res)

    res = execute(res)

    simplification_steps.append(res)

    monom_list = []
    for monom in res.split(' + '):
        multipliers = monom.split("*")
        n = multipliers[0]
        if n in var_set:
            monom_list.append("*".join(multipliers))
        elif n.isdigit() and int(n) % 2 == 1:
            monom_list.append("*".join(multipliers[1:]))

    if monom_list:
        res = ' + '.join(monom_list)
    else:
        res = "0"

    simplification_steps.append(res)
    # show_simplification(simplification_steps)

    return res


def show_simplification(steps):
    print("simplification: ", end="")
    print(" |-> ".join(steps))


def get_all_nulling_vectors(vector):
    # для заданного вектора v находит все возможные векторы z
    # такие что, если v[i] = 1, z[i] = 0; если v[i] = 0, z[i] = 0 or 1
    # кроме нулевого
    # res = [[1,0,1,0], ...] - список списков.
    N = dim(vector)
    n = int(''.join([str(x) for x in vector]), 2)
    res = []
    for i in range(2 ** N):
        if n & i == 0:
            v = list('0' * (N - len(bin(i)[2:])) + bin(i)[2:])
            res.append(list(map(int, v)))
    return res[1:]


# if __name__ == '__main__':
#     TT = set_truth_table()
#     func_vector = TT[-1]
#
#     nulling_vectors = get_all_nulling_vectors(func_vector)
#     print('All nulling_vectors : ', nulling_vectors)
#
#     func_ANF = build_zhegalkin_polynomial(TT[-1], TT[0], TT[1])
#     annig_anf_list = [build_zhegalkin_polynomial(v, TT[0], TT[1]) for v in nulling_vectors]
#
#     print('ANF of func : ', func_ANF)
#     print('ANFs of nulling vectors : ', annig_anf_list)
#
#     for f in annig_anf_list:
#         print("multiplication: %s * %s = %s" % (func_ANF, f, multiply_polynomials(func_ANF, f, TT[0])))
#

# if __name__ == '__main__':
#     res = multiply_polynomials([0, 1, 1, 0], [0, 0, 0, 1])
#     print(res)


if __name__ == "__main__":
    # lfsr_one = LFSR([5, 3])
    # lfsr_two = LFSR([7, 4])
    system_generator = EquationSystemGenerator(ciphera5.LFSR([4, 1]))
    for i, func in enumerate(system_generator.generate_equation_system(20)[0]):
        print("%s\t%s" % (i, func))
