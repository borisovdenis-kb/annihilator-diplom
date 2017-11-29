import math


class EquationSystemGenerator:
    def __init__(self):
        super().__init__()

    


def n_choose_k(n, r):
    f = math.factorial
    return int(f(n) / f(r) / f(n - r))


def count_equation_number_for_mcfsr(*register_length):
    """
    MCFSR - mutually clock-controlled FSR-based stream cipher
    :param register_length: length of registers [count_equation_number(19, 21, 23)]
    :return: number of equation needed to find initial state of each register
    """
    sum_of_length = sum(register_length)
    return sum([n_choose_k(sum_of_length, i) for i in range(1, sum_of_length)])


if __name__ == "__main__":
    print(n_choose_k(5, 2))
    print(count_equation_number_for_mcfsr(19, 22, 23))
    print(2**88)
