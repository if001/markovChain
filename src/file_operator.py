import projectRoot


class FileOperator():
    @classmethod
    def f_open(cls, file_name):
        with open(file_name, "r") as f:
            lines = f.readlines()
        return lines

    @classmethod
    def f_write(cls, file_name, data, spliter=","):
        with open(file_name, 'w') as f:
            f.write(spliter.join(data))
