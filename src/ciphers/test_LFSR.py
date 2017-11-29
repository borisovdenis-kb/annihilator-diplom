from src.ciphers.ciphera5 import LFSR


def popa(n):
    lfsr = LFSR([4, 1], sync_bit=None, init_state=[1, 0, 1, 1])
    for i in range(n):
        print(str(i) + "\t", end="")

    print()
    for i in range(n):
        print(str(lfsr.left_shift()) + "\t", end="")

popa(15)
