import math
import os
from pathlib import Path
from typing import List, Optional, Union


def convert_size(size_bytes: int) -> str:
    # Function to convert bytes to a human-readable format
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    size = round(size_bytes / p, 2)
    return f"{size} {size_name[i]}"


def get_total_file_size(directory_path: Union[str, Path], npz_keys: Optional[List[str]] = None) -> int:
    total_size = 0

    for root, _, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)

            if npz_keys is not None:
                path = Path(file_path)
                if path.stem not in npz_keys or path.suffix != ".npz":
                    continue

            total_size += os.path.getsize(file_path)

    return total_size
