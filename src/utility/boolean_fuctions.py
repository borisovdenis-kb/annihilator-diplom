import sympy
from multipledispatch import dispatch


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
def multiply_polynomials(p1, p2, VARS):
    # функция перемножает многочлены Жег. Р1 и Р2
    # и делает всевозможные упрощения получившегося произведения.
    states = [p2]
    res = []
    s = '(' + p1 + ')*(' + p2 + ')'
    P3 = sympy.expand(s)
    states.append(P3)

    for v in VARS:
        P3 = P3.replace('%s**2' % v, v)

    states.append(P3)

    P3 = P3.__repr__()
    for exp in P3.split(' + '):
        n = exp.split('*')[0]
        if n.isdigit() and int(n) % 2 == 1:
            res.append(''.join(exp[1:]))

    states.append('+'.join(res).replace('', '0'))
    return states


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


if __name__ == '__main__':
    TT = set_truth_table()
    func_vector = TT[-1]

    nulling_vectors = get_all_nulling_vectors(func_vector)
    print('All nulling_vectors : ', nulling_vectors)

    func_ANF = build_zhegalkin_polynomial(TT[-1], TT[0], TT[1])
    ANFs = [build_zhegalkin_polynomial(v, TT[0], TT[1]) for v in nulling_vectors]

    print('ANF of func : ', func_ANF)
    print('ANFs of nulling vectors : ', ANFs)

    Annigs = []
    for f in ANFs:
        Annigs.append(multiply_polynomials(func_ANF, f, TT[0]))
    for anig in Annigs:
        show_annig(func_ANF, anig)

#
# if __name__ == '__main__':
#     res = multiply_polynomials([0, 1, 1, 0], [0, 0, 0, 1])
#     print(res)
