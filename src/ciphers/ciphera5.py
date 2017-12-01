from src.utility.math import boolfunc


class LFSR:
    def __init__(self, config, length, sync_bit_idx=None, init_state=None):
        """ 
        :param config: - многочлен, описывающий конфигурацию РСЛОСа
        т.е. это вектор, задающий позиции отводов в регистре
        пример: [19,18,17,14]
        
        :param length: - длина регистра
        :param sync_bit_idx: - номер синхронизирующего бита
        
        :param init_state: - необязательный параметр, который задает 
        начальное состояние регистра. По умолчанию начальное состояние регистра 
        будет сплошь нулевым, иное состояние устанавливается с помощью вектора
        пример: [0,1,1,0,0...]. Еще проинициализировать можно булевыми многочленами.

        *** наш РСЛОС будет свигаться влево.
        """
        self.length = length
        self.__config = config
        self.__sync_bit_idx = sync_bit_idx

        if not init_state:
            self.__lfsr = [0] * self.length
        else:
            self.__lfsr = init_state

    def blank_shift(self):  # холостой сдвиг
        self.lfsr = [0] + self.lfsr[:-1]

    def bit_left_shift(self):
        output = self.lfsr[0]
        feedback = 0
        for pos in self.__config:  # считаем функцию обратной связи
            feedback ^= self.lfsr[pos - 1]

        self.lfsr = self.lfsr[1:] + [feedback]

        return output

    def func_left_shift(self):
        output = self.lfsr[0]
        feedback = boolfunc.SymbolicBoolFunction(output.var_list, "0")

        for pos in self.__config:           # считаем функцию обратной связи
            feedback = feedback ^ self.lfsr[pos - 1]

        self.lfsr = self.lfsr[1:] + [feedback]

        return output

    @property
    def lfsr(self):
        return self.__lfsr

    @lfsr.setter
    def lfsr(self, vector):
        if len(vector) == self.length:
            self.__lfsr = vector
        else:
            raise ValueError(
                "len of new vector must be the same as len of lfsr"
            )

    def get_sync_bit_value(self):
        return self.__lfsr[self.__sync_bit_idx]

    @property
    def sync_bit_idx(self):
        return self.__sync_bit_idx

    @sync_bit_idx.setter
    def sync_bit_idx(self, idx):
        self.__sync_bit_idx = idx

    def __repr__(self):
        return self.__lfsr.__repr__()


class BitConverter:
    @staticmethod
    def convert_string_to_bit_list(data):
        data = [ord(ch) for ch in data]
        result = []
        for ch in data:
            b = bin(ch)[2:]
            result += [0] * (11 - len(b)) + [int(c) for c in b]
        return result

    @staticmethod
    def convert_bit_list_to_string(bit_list):
        res, n, deg = [], 0, 10
        for i in range(len(bit_list)):
            n += (2 ** deg) * bit_list[i]
            deg -= 1
            if deg < 0:
                res.append(n)
                deg = 10
                n = 0
        return ''.join([chr(x) for x in res])


class CipherA5:
    def __init__(self, key=""):
        """
        :param key: 8-bit string 
        """
        self.r1 = LFSR([19, 18, 17, 14], 7)
        self.r2 = LFSR([22, 21], 9)
        self.r3 = LFSR([23, 22, 21, 8], 9)
        self.key = BitConverter.convert_string_to_bit_list(key)

    def initialize_registers(self):
        for i in range(64):
            self.r1.lfsr[0] ^= self.key[i]
            self.r2.lfsr[0] ^= self.key[i]
            self.r3.lfsr[0] ^= self.key[i]

            self.r1.blank_shift()
            self.r2.blank_shift()
            self.r3.blank_shift()

    def generate_key_stream(self, text):
        """
        :param text: list of bits 
        :return: key_stream: list of bits, 
                 key_stream needed for encryption and decryption
        """
        self.initialize_registers()

        key_stream = []
        for i in range(len(text)):
            r1_output, r2_output, r3_output = 0, 0, 0
            x = self.r1.get_sync_bit_value()
            y = self.r2.get_sync_bit_value()
            z = self.r3.get_sync_bit_value()

            f = x & y | x & z | y & z

            if x == f:
                r1_output = self.r1.bit_left_shift()
            if y == f:
                r2_output = self.r2.bit_left_shift()
            if z == f:
                r3_output = self.r3.bit_left_shift()

            key_stream.append(r1_output ^ r2_output ^ r3_output)

        return key_stream

    def encrypt(self, plain_text=""):
        """
        :param plain_text: string 
        :return: encrypted text (string)
        """
        plain_text = BitConverter.convert_string_to_bit_list(plain_text)
        key_stream = self.generate_key_stream(plain_text)

        cipher_text = []
        for i in range(len(plain_text)):
            cipher_text.append(plain_text[i] ^ key_stream[i])

        return BitConverter.convert_bit_list_to_string(cipher_text)

    def decrypt(self, cipher_text=""):
        """
        :param cipher_text: string
        :return: decrypted text (string)
        """
        cipher_text = BitConverter.convert_string_to_bit_list(cipher_text)
        key_stream = self.generate_key_stream(cipher_text)

        plain_text = []
        for i in range(len(cipher_text)):
            plain_text.append(cipher_text[i] ^ key_stream[i])

        return BitConverter.convert_bit_list_to_string(plain_text)


if __name__ == '__main__':
    plain_text = "deniska"  # input("Enter text: ")
    key = "popapopa"  # input("Enter key (8-bit): ")
    cipher = CipherA5(key=key)

    cipher_text = cipher.encrypt(plain_text=plain_text)
    print(cipher_text)

    plain_text = cipher.decrypt(cipher_text)
    print(plain_text)
