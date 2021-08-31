from typing import Generator


class FileReader:
    def __init__(self):
        self.is_open = False

    def open(self, file_name: str):
        self.f = open(file_name, "r")
        self.is_open = True

    def close(self):
        self.f.close()
        self.is_open = False

    def read_lines(self) -> Generator[str, None, None]:
        if not self.is_open:
            raise RuntimeError("the file has not been opened for reading")

        while True:
            line = self.f.readline()

            if line:
                yield line
            else:
                break
