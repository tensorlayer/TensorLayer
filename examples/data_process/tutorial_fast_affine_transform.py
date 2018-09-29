import tensorlayer as tl
import numpy as np
import time

tl.logging.set_verbosity(tl.logging.DEBUG)

if __name__ == '__main__':

    image = tl.vis.read_image('tiger.jpeg')

    ## apply transforms one-by-one is very SLOW !
    st = time.time()
    for _ in range(100):
        xx = tl.prepro.rotation(image, rg=20, is_random=False)
        xx = tl.prepro.shift(xx, wrg=0.2, hrg=0.2, is_random=False)
        xx = tl.prepro.flip_axis(xx, axis=1, is_random=False)
        xx = tl.prepro.zoom(xx, zoom_range=(0.8, 1.5), is_random=False)
        xx = tl.prepro.shear(xx, intensity=0.2, is_random=False)
    print("apply transforms one-by-ine took %fs" % ((time.time()-st) / 100))
    tl.vis.save_image(xx, '_result_slow.png')

    ## apply all transforms once is very FAST !
    st = time.time()
    h, w, _ = image.shape
    for _ in range(100):
        # 1. combine the combined affine transform matrix
        M_rotate = tl.prepro.affine_rotation_matrix(rg=20, is_random=False)
        M_flip = tl.prepro.affine_horizontal_flip_matrix(is_random=False)
        M_shift = tl.prepro.affine_shift_matrix(wrg=0.2, hrg=0.2, h=h, w=w, is_random=False)
        M_shear = tl.prepro.affine_shear_matrix(intensity=0.2, is_random=False)
        M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.8, 1.5), is_random=False)
        M_combined = M_rotate.dot(M_shift).dot(M_flip).dot(M_zoom).dot(M_shear) # this means rotation is the first transformation
        # 2. transfrom the matrix from Cartesian coordinate (the origin in the middle of image)
        # to Image coordinate (the origin on the top-left of image)
        transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, h, w)
        # 3. then we can transfrom the image once for all transformations
        result = tl.prepro.affine_transfrom(image, transform_matrix) # apply_affine_transform
    print("apply all transforms once took %fs" % ((time.time()-st) / 100))
    tl.vis.save_image(result, '_result_fast.png')
