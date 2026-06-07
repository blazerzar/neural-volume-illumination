"""
Reads quality experiment declarations and moves downloaded image ZIPs
from ~/Downloads into the data/images/ directory.
"""

import json
import logging
import os
import shutil

DOWNLOAD_DIR = os.path.expanduser('~/Downloads')
EXPERIMENTS_DIR = 'evaluation/experiments/quality'
IMAGES_DIR = 'data/images'

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
        if experiment.get('type') != 'quality':
            continue

        download_name = f'{name}.zip'
        dest_path = os.path.join(IMAGES_DIR, download_name)

        if os.path.exists(dest_path):
            logging.info(f'[{name}] Image ZIP {dest_path} already exists')
            continue

        download_path = os.path.join(DOWNLOAD_DIR, download_name)
        if os.path.exists(download_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(download_path, dest_path)
            logging.info(f'[{name}] Moved {download_path} -> {dest_path}')
        else:
            logging.warning(
                f'[{name}] Image ZIP {download_name} not found in '
                f'{dest_path} or {download_path}'
            )


if __name__ == '__main__':
    main()
