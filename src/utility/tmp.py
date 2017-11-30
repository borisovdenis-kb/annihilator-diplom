from sympy import poly, symbols, Poly


def add_poly():
    var_list = [symbols('x'), symbols('y')]

    p1 = Poly('x', var_list[0])
    p2 = Poly('y', var_list[1])

    res = p1.add(p2)

    return res


if __name__ == "__main__":
    print(add_poly())
