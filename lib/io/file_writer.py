from typing import Any

class FileWriter:

    def __init__(self):
        self.is_open = False

    def open(self, file_name: str):
        self.f = open(file_name, 'w')
        self.is_open = True

    def close(self):
        self.f.close()
        self.is_open = False

    def write(self, obj: Any):
        if not self.is_open:
            raise RuntimeError('the file has not been opened for writing')

        self.f.write(str(obj))

