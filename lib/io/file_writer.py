from typing import Any
import os


class FileWriter:
    def __init__(self):
        self.is_open = False
        self.swap_file = None
        self.file_name = None
        self.f = None

    def open(self, file_name: str) -> None:
        self.file_name = file_name
        self.swap_file = self._swap_file(file_name)
        self.f = open(self.swap_file, "w")
        self.is_open = True

    def close(self) -> None:
        self.f.close()
        
        # Replace file with swap file and remove swap file
        os.replace(self.swap_file, self.file_name)
        
        self.swap_file = None
        self.file_name = None
        self.f = None
        self.is_open = False

    def write(self, obj: Any) -> None:
        if not self.is_open:
            raise RuntimeError("the file has not been opened for writing")

        self.f.write(str(obj))
    
    def _swap_file(self, file_name: str) -> str:
        return file_name + '.swp'
