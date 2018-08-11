import plyvel
import numpy as np


class ProbKvs():
    def __init__(self, save_name="prob.ldb"):
        self.my_db = plyvel.DB(save_name, create_if_missing=True)

    def __del__(self):
        self.my_db.close()

    def put(self, key, value):
        """
        key: type:string
        value: type: list object
        """
        self.my_db.put(self.__u(key), self.__b(value))

    def get(self, key):
        """
        return int array
        """
        value = self.my_db.get(self.__u(key))

        if value is not None:
            value = [x for x in value]
        return value

    def delete(self, key):
        self.my_db.delete(self.__u(key))

    def __u(self, st):
        return st.encode('utf-8')

    def __b(self, byte_obj):
        return bytes(byte_obj)


def main():
    i2k = ProbKvs()
    i2k.put("あい", [4])
    print(i2k.get("あい"))
    print(type(i2k.get("あい")))


if __name__ == "__main__":
    main()
