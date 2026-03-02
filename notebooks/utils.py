import zipfile

import numpy as np
from tqdm import tqdm

# Number of float32 values per pixel (x, y, z, azimuth, elevation, r, g, b)
PIXEL_VALUES = 8


def read_ground_truth_zip(file_path: str, verbose=False) -> list[np.ndarray]:
    arrays = []
    with zipfile.ZipFile(file_path, mode='r') as archive:
        files = sorted(archive.namelist())

        if verbose:
            files = tqdm(files, desc='Reading ground truth zip')
        for file in files:
            if file.endswith('.bin'):
                buffer_bytes = archive.read(file)
                array = np.frombuffer(buffer_bytes, dtype=np.float32)
                array = array.reshape(-1, PIXEL_VALUES)
                arrays.append(array)

    return arrays
