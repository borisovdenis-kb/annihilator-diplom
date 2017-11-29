from sympy import *
from Zhegalkin import *


def dim(vector):
    return len(vector)


def showAnnig(func, annig):
    # функция демонстрирует работу аннигилятора для функции func
    print()
    print('Annig : %s' % annig[0])
    print('(%s)*(%s) = %s = %s = %s' % (func, annig[0], annig[1], annig[2], annig[3]))


def mulPoly(P1, P2, VARS):
    # функция перемножает многочлены Жег. Р1 и Р2
    # и делает всевозможные упрощения получившегося произведения.
    states = [P2]
    res = []
    s = '(' + P1 + ')*(' + P2 + ')'
    P3 = expand(s)
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


def allNullingVectors(vector):
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
    TT = setTruthTable()
    func_vector = TT[-1]

    nulling_vectors = allNullingVectors(func_vector)
    print('All nulling_vectors : ', nulling_vectors)

    func_ANF = buildZhegalkin(TT[-1], TT[0], TT[1])
    ANFs = [buildZhegalkin(v, TT[0], TT[1]) for v in nulling_vectors]

    print('ANF of func : ', func_ANF)
    print('ANFs of nulling vectors : ', ANFs)

    Annigs = []
    for f in ANFs:
        Annigs.append(mulPoly(func_ANF, f, TT[0]))
    for anig in Annigs:
        showAnnig(func_ANF, anig)
