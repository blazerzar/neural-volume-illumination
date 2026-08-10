import os
import zipfile

IMAGES_DIR = os.path.join('data', 'images')
OUT_DIR = 'images'

IMAGES = {
    'front': [
        ('bonsai', 200, 3),
        ('chameleon', 200, 2),
        ('silicium', 1000, 2),
    ],
    'turntable': [
        ('chameleon', 200, 3, 60),
        ('chameleon', 200, 3, 240),
        ('chameleon', 200, 3, 300),
    ],
    'good': [
        ('bonsai', 80, 3),
        ('csafe_heptane', 80, 3),
        ('frog', 200, 2),
        ('marmoset_neurons', 80, 2),
        ('miranda', 1000, 1),
        ('silicium', 200, 3),
        ('vismale', 1000, 1),
    ],
    'bad': [
        ('bonsai', 1000, 3),
        ('chameleon', 1000, 3),
        ('frog', 80, 3),
        ('marmoset_neurons', 200, 2),
        ('miranda', 1000, 3),
        ('mri_ventricles', 200, 2),
        ('silicium', 1000, 3),
        ('vismale', 1000, 3),
    ],
}


def main():
    for experiment, images in IMAGES.items():
        if not os.path.exists(os.path.join(OUT_DIR, experiment)):
            os.makedirs(os.path.join(OUT_DIR, experiment))

        for volume, ext, tf, *angle in images:
            for method, m in [('path_tracing', 'pt'), ('neural_render', 'nr')]:
                for mode in ['global', 'indirect']:
                    zip_path = create_zip_path(method, volume, ext, tf, *angle)
                    file_name = create_file_name(volume, ext, tf, mode, m, *angle)
                    out_path = os.path.join(OUT_DIR, experiment, file_name)
                    extract_file(zip_path, mode, out_path, experiment)


def create_file_name(volume, ext, tf, mode, method, *angle):
    parts = [str(x) for x in (volume, ext, tf, *angle, mode, method)]
    return '_'.join(parts) + '.png'


def create_zip_path(method, volume, ext, tf, angle=None):
    ending = f'{volume}_{ext}_{tf}.zip'
    if angle is None:
        return os.path.join(IMAGES_DIR, f'quality_front_{method}_{ending}')
    else:
        return os.path.join(IMAGES_DIR, f'quality_turntable_{angle}_{method}_{ending}')


def extract_file(zip_path, mode, out_path, experiment):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        files = sorted(
            [
                (int(f.split('/')[2][:-4].split('_')[1]), f)
                for f in zip_ref.namelist()
                if f'0/{mode}' in f
            ]
        )
        file = (
            files[-1]
            if experiment == 'front' or experiment == 'turntable'
            else files[1]
        )
        with zip_ref.open(file[1]) as img_file, open(out_path, 'wb') as out_file:
            out_file.write(img_file.read())


if __name__ == '__main__':
    main()
