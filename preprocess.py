"""Preprocess image data for Dogs vs. Cats competition"""
import glob
from multiprocessing import Process
import os
import re

import numpy as np
import PIL
from PIL import Image
from progressbar import ProgressBar

SIZE = 224  # for ImageNet models compatibility
TEST_DIR = 'data/test/'
TRAIN_DIR = 'data/train/'
BASE_DIR = '.'


def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image

    Normalizes each color channel to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    # Convert to a NumPy array
    # The array will be transposed, but it doesn't matter, so we leave it
    img_nrm = np.asarray(img).astype(float)
    clrs = img_nrm.shape[2]

    # Normalize the image to [0,1]
    for clr in range(clrs):
        img_p = img_nrm[..., clr] / 255
        img_p -= img_p.mean()
        img_p /= img_p.std()
        scale = np.max([np.abs(np.percentile(img_p, 1.0)),
                        np.abs(np.percentile(img_p, 99.0))])
        img_p = img_p / scale
        img_p = np.clip(img_p, -1.0, 1.0)
        img_p = (img_p + 1.0) / 2.0
        img_nrm[..., clr] = img_p

    # Convert back to PIL image
    img_nrm = (img_nrm * 255 + 0.5).astype(np.uint8)

    return Image.fromarray(img_nrm)


def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(paths, out_dir):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir

    """
    n_paths = len(paths)
    with ProgressBar(max_value=n_paths) as progress:
        for count, path in enumerate(paths):
            # if count % 100 == 0:
            #     print(path)
            img = Image.open(path)
            img_nrm = norm_image(img)
            img_res = resize_image(img_nrm, SIZE)
            basename = os.path.basename(path)
            path_out = os.path.join(out_dir, basename)
            img_res.save(path_out)
            progress.update(count)


def main():
    """Main program for running from command line"""

    # Get the paths to all the image files
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key)
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key)
    # train_all = train_cats + train_dogs

    test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')), key=natural_key)

    # Make the output directories
    base_out = os.path.join(BASE_DIR, 'data{}'.format(SIZE))
    #!!!
    # if os.path.isdir(base_out):
    #     raise RuntimeError('Output directory {} already exists'.format(base_out))
    #!!!
    train_dir_out = os.path.join(base_out, 'train')
    test_dir_out = os.path.join(base_out, 'test')
    os.makedirs(train_dir_out, exist_ok=True)
    os.makedirs(test_dir_out, exist_ok=True)

    # Preprocess the training files
    procs = dict()
    procs[1] = Process(target=prep_images, args=(train_cats, train_dir_out, ))
    procs[1].start()
    procs[2] = Process(target=prep_images, args=(train_dogs, train_dir_out, ))
    procs[2].start()
    procs[3] = Process(target=prep_images, args=(test_all, test_dir_out, ))
    procs[3].start()

    procs[1].join()
    procs[2].join()
    procs[3].join()


if __name__ == '__main__':
    main()
