import numpy as np
import sys
import io
from mrcnn.config import Config
import unittest


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = Config()

    def test_config_values(self):
        self.assertEqual(self.config.GPU_COUNT, 1)
        self.assertEqual(self.config.IMAGES_PER_GPU, 2)
        self.assertEqual(self.config.STEPS_PER_EPOCH, 1000)
        self.assertEqual(self.config.BACKBONE, "resnet101")
    
    def test_config_value_types(self):
        self.assertIsInstance(self.config.BACKBONE, str)
        self.assertIsInstance(self.config.BACKBONE_STRIDES, list)
        self.assertIsInstance(self.config.MEAN_PIXEL, np.ndarray)
        self.assertIsInstance(self.config.LOSS_WEIGHTS, dict)
        self.assertIsInstance(self.config.USE_RPN_ROIS, bool)
        self.assertIsInstance(self.config.TRAIN_BN, bool)
        self.assertIsInstance(self.config.IMAGE_META_SIZE, int)
    
    def test_config_value_shapes(self):
        self.assertEqual(len(self.config.BACKBONE_STRIDES), 5)
        self.assertEqual(len(self.config.MINI_MASK_SHAPE), 2)
        self.assertEqual(self.config.MEAN_PIXEL.shape, (3,))
        self.assertEqual(self.config.IMAGE_SHAPE.shape, (3,))
    
    def test_config_value_calc(self):
        self.assertEqual(self.config.BATCH_SIZE , self.config.IMAGES_PER_GPU * self.config.GPU_COUNT)
        
    def test_display_func(self):
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        self.config.display()
        self.assertTrue('Configurations:' in capturedOutput.getvalue())
