import sys
import io
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from src.mainloop import run

class Tee(io.TextIOBase):
    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()


if __name__ == "__main__":
    log_name = f"./logs/run_{datetime.now():%Y%m%d_%H%M%S}.log"
    with open(log_name, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
        try:
            run()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
