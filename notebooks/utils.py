import zipfile

import numpy as np
from tqdm import tqdm


def read_ground_truth_zip(file_path: str, verbose=False) -> list[np.ndarray]:
    arrays = []
    with zipfile.ZipFile(file_path, mode='r') as archive:
        files = sorted(archive.namelist())

        if verbose:
            files = tqdm(files, desc='Reading ground truth zip')
        for file in files:
            if file.endswith('.bin'):
                buffer_bytes = archive.read(file)
                arrays.append(np.frombuffer(buffer_bytes, dtype=np.float32))

    return arrays
