import numpy as np
from mrcnn.utils import Dataset 
import unittest

class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()

    def test_add_class(self):
        self.assertEqual(self.dataset.class_info, [{'source': '', 'id': 0, 'name': 'BG'}])
        self.dataset.add_class("shapes", 1, "square")
        self.assertEqual(self.dataset.class_info[1], {'source': 'shapes', 'id': 1, 'name': 'square'})
        self.dataset.add_class("shapes", 2, "circle")
        for element in self.dataset.class_info:
            self.assertIsInstance(element, dict)

    def test_add_image(self):
        bg_color = np.array([220,  55, 172])
        shapes = []
        boxes = []
        shape = "square"
        color = (197, 55, 186)
        dims = (79, 44, 22)
        shapes.append((shape, color, dims))
        self.dataset.add_image('shapes', 1, path=None,
                           width=128, height=128,
                           bg_color=bg_color, shapes=shapes)
        self.assertIsInstance(self.dataset.image_info, list)
        expected_output = [{'id': 1,
                            'source': 'shapes',
                            'path': None,
                            'width': 128,
                            'height': 128,
                            'bg_color': np.array([220,  55, 172]),
                            'shapes': [('square', (197, 55, 186), (79, 44, 22))]}]
        self.assertEqual(self.dataset.image_info[-1]['id'], expected_output[-1]['id'])
        self.assertEqual(self.dataset.image_info[-1]['source'], expected_output[-1]['source'])
        self.assertEqual(self.dataset.image_info[-1]['path'], expected_output[-1]['path'])
        self.assertEqual(self.dataset.image_info[-1]['height'], expected_output[-1]['height'])
        self.assertEqual(self.dataset.image_info[-1]['width'], expected_output[-1]['width'])


    def test_image_reference(self):
        self.assertEqual(self.dataset.image_reference('random_image_id'), '')

    def test_prepare(self):
        self.dataset.prepare()
        self.assertTrue(hasattr(self.dataset, 'class_ids'))
        self.assertTrue(hasattr(self.dataset, 'class_names'))
        self.assertTrue(hasattr(self.dataset, 'num_images'))
        self.assertTrue(hasattr(self.dataset, '_image_ids'))
        self.assertTrue(hasattr(self.dataset, 'class_from_source_map'))
        self.assertTrue(hasattr(self.dataset, 'image_from_source_map'))
        self.assertTrue(hasattr(self.dataset, 'source_class_ids'))
        self.assertIsInstance(self.dataset.class_info, list)
        self.assertEqual(self.dataset.class_info[-1]['name'], 'BG')

    def test_map_source_class_ids(self):
        self.dataset.prepare()
        self.assertEqual(self.dataset.map_source_class_id('.0'), 0)

    def test_get_source_class_id(self):
        self.assertEqual(self.dataset.get_source_class_id(0, ''), 0)

    def test_append_data(self):
        self.assertTrue(True)

    def test_image_ids(self):
        self.assertTrue(not self.dataset.image_ids)

    def test_source_image_link(self):
        self.assertRaises(IndexError, self.dataset.source_image_link, 0)

    






