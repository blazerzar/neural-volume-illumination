import json
import zipfile

import numpy as np
import torch
from tqdm import tqdm

# Number of float32 values per pixel (x, y, z, azimuth, elevation, r, g, b)
PIXEL_VALUES = 8


def read_ground_truth_zip(
    file_path: str, verbose=False
) -> tuple[list[np.ndarray], dict]:
    """Read the ground truth data from a zip file.

    Parameters:
        - file_path: Path to the zip file containing the ground truth data.
        - verbose: Whether to show a progress bar while reading the zip file.

    Returns:
        - List of numpy arrays, one for each frame, containing the pixel data.
        - Dictionary of parameters read from the parameters.json file in the zip.
    """
    arrays = []
    with zipfile.ZipFile(file_path, mode='r') as archive:
        files = sorted(archive.namelist())
        parameters = {}

        for file in tqdm(files, disable=not verbose, desc='Reading ground truth zip'):
            if file.endswith('.bin'):
                buffer_bytes = archive.read(file)
                array = np.frombuffer(buffer_bytes, dtype=np.float32)
                array = array.reshape(-1, PIXEL_VALUES)
                arrays.append(array)
            elif file == 'parameters.json':
                buffer_bytes = archive.read(file)
                parameters = json.loads(buffer_bytes.decode('utf-8'))

    return arrays, parameters


def preprocess_data(
    data: list[np.ndarray], device: torch.device, verbose=False
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[np.ndarray]]:
    """Remove empty pixels and convert to PyTorch tensors.

    Parameters:
        - data: List of numpy arrays returned by read_ground_truth_zip.
        - device: The device to move the tensors to.
        - verbose: Whether to show progress bar.

    Returns:
        - List of feature tensors (x, y, z, azimuth, elevation) for each frame.
        - List of indirect radiance tensors (r, g, b) for each frame.
        - List of boolean masks indicating valid pixels for each frame.
    """
    Xs, ys, masks = [], [], []
    for frame in tqdm(data, disable=not verbose, desc='Preprocessing data'):
        non_zero_mask = (frame != 0).any(axis=1)
        non_nan_mask = ~np.isnan(frame).any(axis=1)
        valid_mask = non_zero_mask & non_nan_mask
        filtered_frame = frame[valid_mask]
        tensor_frame = torch.from_numpy(filtered_frame).float().to(device)

        # Normalize azimuth and elevation to [0, 1]
        tensor_frame[:, 3] = (tensor_frame[:, 3] + np.pi) / (2.01 * np.pi)
        tensor_frame[:, 4] = tensor_frame[:, 4] / np.pi

        Xs.append(tensor_frame[:, :5])
        ys.append(tensor_frame[:, 5:])
        masks.append(valid_mask)

    return Xs, ys, masks


def create_image(values: torch.Tensor, mask: np.ndarray, resolution: int) -> np.ndarray:
    """Create an image from the given values and mask. The data can be either
    the ground truth or the predicted values.

    Parameters:
        - values: Tensor of shape (num_valid_pixels, 3) containing the RGB values.
        - mask: Boolean array indicating valid pixels, returned by preprocess_data.
        - resolution: The width and height of the output images.

    Returns:
        - Numpy array of shape (resolution, resolution, 3).
    """
    image = np.zeros((resolution, resolution, 3))
    image_flat = image.reshape(-1, 3)
    values_cpu = values.detach().cpu().numpy()

    indices = np.where(mask)[0]
    image_flat[indices] = values_cpu
    return image


def export_state_dict(model, model_args, file_name=None, buffer=None):
    """Export the model's state_dict as a zip file for browser inference.

    Parameters:
        - model: The PyTorch model to export.
        - model_args: Dictionary of used model arguments
        - file_name: The name of the output zip file.

    Returns:

    """
    metadata = {'model_args': model_args}
    dest = buffer if file_name is None else file_name

    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            data = t.view(-1).numpy().tobytes()
            name = name.replace('.', '_')
            file = f'{name}.bin'

            zf.writestr(file, data)

            metadata[name] = {
                'file': file,
                'size': list(tensor.size()),
                'dtype': str(tensor.dtype),
                'num_elements': tensor.numel(),
                'bytes_per_element': tensor.element_size(),
                'total_bytes': tensor.numel() * tensor.element_size(),
            }

        zf.writestr('metadata.json', json.dumps(metadata, indent=4))
