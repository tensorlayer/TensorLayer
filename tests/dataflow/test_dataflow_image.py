#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorlayer as tl

from tests.utils import CustomTestCase


class Dataflow_Image_Test(CustomTestCase):

    @classmethod
    def setUpClass(self):
        self.input_shape = [1, 100, 100, 3]
        self.input_layer = tl.layers.Input(self.input_shape, name='input_layer')
        self.input_shape_1 = [100, 100, 3]
        self.input_layer_1 = tl.layers.Input(self.input_shape_1, name='input_layer_1')

        self.centralcrop_1 = tl.dataflow.image.CentralCrop(self.input_layer, central_fraction=0.5)
        self.centralcrop_2 = tl.dataflow.image.CentralCrop(self.input_layer, size=60)

        self.hsvtorgb = tl.dataflow.image.HsvToRgb(self.input_layer)

        self.adjustbrightness = tl.dataflow.image.AdjustBrightness(self.input_layer, factor=0.5)
        self.adjustconstrast = tl.dataflow.image.AdjustContrast(self.input_layer, factor=0.5)
        self.adjusthue = tl.dataflow.image.AdjustHue(self.input_layer, factor=0.5)
        self.adjustsaturation = tl.dataflow.image.AdjustSaturation(self.input_layer, factor=0.5)

        self.crop = tl.dataflow.image.Crop(
            self.input_layer, offset_height=20, offset_width=20, target_height=60, target_width=60
        )

        self.fliphorizontal = tl.dataflow.image.FlipHorizontal(self.input_layer)
        self.flipvertical = tl.dataflow.image.FlipVertical(self.input_layer)

        self.rgbtogray = tl.dataflow.image.RgbToGray(self.input_layer)
        self.graytorgb = tl.dataflow.image.GrayToRgb(self.rgbtogray)

        self.padtoboundingbox = tl.dataflow.image.PadToBoundingbox(
            self.input_layer, offset_height=20, offset_width=20, target_height=150, target_width=150
        )

        self.pad_1 = tl.dataflow.image.Pad(self.input_layer, padding=10, padding_value=1, mode='constant')
        self.pad_2 = tl.dataflow.image.Pad(self.input_layer, padding=(10, 10), mode='REFLECT')
        self.pad_3 = tl.dataflow.image.Pad(self.input_layer, padding=(10, 20, 30, 40), mode='SYMMETRIC')

        self.standardization_1 = tl.dataflow.image.Standardization(
            self.input_layer, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        self.standardization_2 = tl.dataflow.image.Standardization(self.input_layer, channel_mode=False)
        self.standardization_3 = tl.dataflow.image.Standardization(self.input_layer, channel_mode=True)

        self.randombrightness = tl.dataflow.image.RandomBrightness(self.input_layer, factor=0.5)
        self.randomcontrast = tl.dataflow.image.RandomContrast(self.input_layer, lower=0.2, upper=0.5)
        self.randomhue = tl.dataflow.image.RandomHue(self.input_layer, factor=0.5)
        self.randomsaturation = tl.dataflow.image.RandomSaturation(self.input_layer, lower=0.2, upper=0.5)

        self.randomcrop_1 = tl.dataflow.image.RandomCrop(self.input_layer, size=50)
        self.randomcrop_2 = tl.dataflow.image.RandomCrop(self.input_layer, size=(50, 60))

        self.resize_1 = tl.dataflow.image.Resize(
            self.input_layer, size=46, method='bilinear', preserve_aspect_ratio=False, antialias=True
        )

        self.resize_2 = tl.dataflow.image.Resize(
            self.input_layer, size=(32, 45), method='bilinear', preserve_aspect_ratio=True, antialias=False
        )

        self.croporpad = tl.dataflow.image.CropOrPad(self.input_layer, target_height=50, target_width=150)
        self.resizeandpad = tl.dataflow.image.ResizeAndPad(
            self.input_layer, target_height=50, target_width=150, method='bilinear'
        )
        self.rgbtohsv = tl.dataflow.image.RgbToHsv(self.input_layer)
        self.transpose = tl.dataflow.image.Transpose(self.input_layer, order=(3, 2, 1, 0))
        self.randomrotation = tl.dataflow.image.RandomRotation(
            self.input_layer_1, degrees=60, fill_mode='nearest', fill_value=1
        )
        self.randomshift_1 = tl.dataflow.image.RandomShift(
            self.input_layer_1, shift=0.5, fill_mode='nearest', fill_value=0
        )
        self.randomshift_2 = tl.dataflow.image.RandomShift(
            self.input_layer_1, shift=(0.5, 0.4), fill_mode='nearest', fill_value=0
        )

        self.randomshear = tl.dataflow.image.RandomShear(
            self.input_layer_1, degree=30, fill_mode='nearest', fill_value=1
        )

        self.randomzoom_1 = tl.dataflow.image.RandomZoom(
            self.input_layer_1, zoom_range=0.5, fill_mode='nearest', fill_value=1
        )
        self.randomzoom_2 = tl.dataflow.image.RandomZoom(
            self.input_layer_1, zoom_range=(0.5, 0.4), fill_mode='nearest', fill_value=1
        )

        self.rescale = tl.dataflow.image.Rescale(self.input_layer, scale=3, offset=4)
        self.randomflipvertical = tl.dataflow.image.RandomFlipVertical(self.input_layer)
        self.randomfliphorizontal = tl.dataflow.image.RandomFlipHorizontal(self.input_layer)
        self.hwc2chw = tl.dataflow.image.HWC2CHW(self.input_layer)
        self.chw2hwc = tl.dataflow.image.CHW2HWC(self.hwc2chw)

    @classmethod
    def tearDownClass(self):
        pass

    def test_centralcrop_1(self):

        self.assertEqual(tl.get_tensor_shape(self.centralcrop_1), [1, 50, 50, 3])

    def test_centralcrop_2(self):

        self.assertEqual(tl.get_tensor_shape(self.centralcrop_2), [1, 60, 60, 3])

    def test_hsvtorgb(self):

        self.assertEqual(tl.get_tensor_shape(self.hsvtorgb), [1, 100, 100, 3])

    def test_adjustbrightness(self):

        self.assertEqual(tl.get_tensor_shape(self.adjustbrightness), [1, 100, 100, 3])

    def test_adjustconstrast(self):

        self.assertEqual(tl.get_tensor_shape(self.adjustconstrast), [1, 100, 100, 3])

    def test_adjusthue(self):

        self.assertEqual(tl.get_tensor_shape(self.adjusthue), [1, 100, 100, 3])

    def test_adjustsaturation(self):

        self.assertEqual(tl.get_tensor_shape(self.adjustsaturation), [1, 100, 100, 3])

    def test_crop(self):

        self.assertEqual(tl.get_tensor_shape(self.crop), [1, 60, 60, 3])

    def test_fliphorizontal(self):

        self.assertEqual(tl.get_tensor_shape(self.fliphorizontal), [1, 100, 100, 3])

    def test_flipvertical(self):

        self.assertEqual(tl.get_tensor_shape(self.flipvertical), [1, 100, 100, 3])

    def test_rgbtogray(self):

        self.assertEqual(tl.get_tensor_shape(self.rgbtogray), [1, 100, 100, 1])

    def test_graytorgb(self):

        self.assertEqual(tl.get_tensor_shape(self.graytorgb), [1, 100, 100, 3])

    def test_padtoboundingbox(self):

        self.assertEqual(tl.get_tensor_shape(self.padtoboundingbox), [1, 150, 150, 3])

    def test_pad_1(self):

        self.assertEqual(tl.get_tensor_shape(self.pad_1), [1, 120, 120, 3])

    def test_pad_2(self):

        self.assertEqual(tl.get_tensor_shape(self.pad_2), [1, 120, 120, 3])

    def test_pad_3(self):

        self.assertEqual(tl.get_tensor_shape(self.pad_3), [1, 130, 170, 3])

    def test_standardization_1(self):

        self.assertEqual(tl.get_tensor_shape(self.standardization_1), [1, 100, 100, 3])

    def test_standardization_2(self):

        self.assertEqual(tl.get_tensor_shape(self.standardization_2), [1, 100, 100, 3])

    def test_standardization_3(self):

        self.assertEqual(tl.get_tensor_shape(self.standardization_3), [1, 100, 100, 3])

    def test_randomcontrast(self):

        self.assertEqual(tl.get_tensor_shape(self.randomcontrast), [1, 100, 100, 3])

    def test_randomhue(self):

        self.assertEqual(tl.get_tensor_shape(self.randomhue), [1, 100, 100, 3])

    def test_randomsaturation(self):

        self.assertEqual(tl.get_tensor_shape(self.randomsaturation), [1, 100, 100, 3])

    def test_randomcrop_1(self):

        self.assertEqual(tl.get_tensor_shape(self.randomcrop_1), [1, 50, 50, 3])

    def test_randomcrop_2(self):

        self.assertEqual(tl.get_tensor_shape(self.randomcrop_2), [1, 50, 60, 3])

    def test_resize_1(self):

        self.assertEqual(tl.get_tensor_shape(self.resize_1), [1, 46, 46, 3])

    def test_resize_2(self):

        self.assertEqual(tl.get_tensor_shape(self.resize_2), [1, 32, 32, 3])

    def test_croporpad(self):

        self.assertEqual(tl.get_tensor_shape(self.croporpad), [1, 50, 150, 3])

    def test_resizeandpad(self):

        self.assertEqual(tl.get_tensor_shape(self.resizeandpad), [1, 50, 150, 3])

    def test_rgbtohsv(self):

        self.assertEqual(tl.get_tensor_shape(self.rgbtohsv), [1, 100, 100, 3])

    def test_transpose(self):

        self.assertEqual(tl.get_tensor_shape(self.transpose), [3, 100, 100, 1])

    def test_randomrotation(self):

        self.assertEqual(tl.get_tensor_shape(self.randomrotation), [100, 100, 3])

    def test_randomshift_1(self):

        self.assertEqual(tl.get_tensor_shape(self.randomshift_1), [100, 100, 3])

    def test_randomshift_2(self):

        self.assertEqual(tl.get_tensor_shape(self.randomshift_2), [100, 100, 3])

    def test_randoshear(self):

        self.assertEqual(tl.get_tensor_shape(self.randomshear), [100, 100, 3])

    def test_randomzoom_1(self):

        self.assertEqual(tl.get_tensor_shape(self.randomzoom_1), [100, 100, 3])

    def test_randomzoom_2(self):

        self.assertEqual(tl.get_tensor_shape(self.randomzoom_2), [100, 100, 3])

    def test_rescale(self):

        self.assertEqual(tl.get_tensor_shape(self.rescale), [1, 100, 100, 3])

    def test_randomflipvertical(self):

        self.assertEqual(tl.get_tensor_shape(self.randomflipvertical), [1, 100, 100, 3])

    def test_randomfliphorizontal(self):

        self.assertEqual(tl.get_tensor_shape(self.randomfliphorizontal), [1, 100, 100, 3])

    def test_hwc2chw(self):

        self.assertEqual(tl.get_tensor_shape(self.hwc2chw), [1, 3, 100, 100])

    def test_chw2hwc(self):

        self.assertEqual(tl.get_tensor_shape(self.chw2hwc), [1, 100, 100, 3])


if __name__ == '__main__':

    tl.logging.set_verbosity(tl.logging.DEBUG)

    unittest.main()
