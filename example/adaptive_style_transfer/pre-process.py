# A Script to Pre-process WikiArt dataset
# This script helps to discard the "bad" images
# which cannot be well used by the training method.

from __future__ import print_function

from datetime import datetime
from os import listdir, remove
from os.path import join
from scipy.misc import imread, imresize

PATH = '/root/even/dataset/wiki_all_images'


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def main(dir_path):

    paths = list_images(dir_path)

    print('\norigin files number: %d\n' % len(paths))

    num_delete = 0

    for path in paths:

        try:
            image = imread(path, mode='RGB')
        except IOError:
            num_delete += 1
            print('Cant read this file, will delete it')
            remove(path)

        if len(image.shape) != 3 or image.shape[2] != 3:
            num_delete += 1
            remove(path)
            print('\nimage.shape:', image.shape, ' Remove image <%s>\n' % path)
        else:
            height, width, _ = image.shape

            if height < width:
                new_height = 512
                new_width = int(width * new_height / height)
            else:
                new_width = 512
                new_height = int(height * new_width / width)

            try:
                image = imresize(image, [new_height, new_width], interp='nearest')
            except:
                print('Cant resize this file, will delete it')
                num_delete += 1
                remove(path)

    print('\n\ndelete %d files! Current number of files: %d\n\n' % (num_delete, len(paths) - num_delete))


if __name__ == '__main__':
    t0 = datetime.now()

    main(PATH)

    dt = datetime.now() - t0
    print('elapsed time:', dt, '\n')
