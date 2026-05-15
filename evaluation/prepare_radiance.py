"""
Reads experiment declarations and prepares radiance ground truth ZIP files.
Checks all experiment JSONs for 'radiance' keys and ensures the referenced
ZIP files are available in the data directory. If found in the download
directory, moves them into place.
"""

import json
import logging
import os
import shutil

DOWNLOAD_DIR = os.path.expanduser('~/Downloads')
EXPERIMENTS_DIR = 'evaluation/experiments'

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def find_experiments(root):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.json'):
                yield os.path.join(dirpath, filename)


def main():
    for experiment_path in find_experiments(EXPERIMENTS_DIR):
        with open(experiment_path) as f:
            experiment = json.load(f)

        name = experiment.get('name', experiment_path)
        radiance = experiment.get('radiance')

        if not radiance:
            continue

        if os.path.exists(radiance):
            logging.info(f'[{name}] Radiance file {radiance} found')
            continue

        download_path = os.path.join(DOWNLOAD_DIR, os.path.basename(radiance))
        if os.path.exists(download_path):
            os.makedirs(os.path.dirname(radiance), exist_ok=True)
            shutil.move(download_path, radiance)
            logging.info(f'[{name}] Moved {download_path} -> {radiance}')
        else:
            logging.warning(
                f'[{name}] Radiance file {radiance} not found in '
                f'{radiance} or {download_path}'
            )


if __name__ == '__main__':
    main()
