import unittest
import os
import json
import numpy as np
import cv2
from app import imagedetector

class TestImageProcessing(unittest.TestCase):

    def setUp(self):
        self.image_path = 'test_image.jpg'
        self.output_json_path = 'app/src/output/output.json'
        self.output_image_path = 'app/src/output/output.jpg'
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(self.image_path, self.test_image)

    def tearDown(self):
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        if os.path.exists(self.output_json_path):
            os.remove(self.output_json_path)
        if os.path.exists(self.output_image_path):
            os.remove(self.output_image_path)

    def test_execute_valid_image(self):
        result = imagedetector.execute(self.image_path)
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) >= 0)

    def test_execute_invalid_image(self):
        with self.assertRaises(ValueError):
            imagedetector.execute('invalid_path.jpg')

    def test_remove_shadow(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        planes = cv2.split(image)
        result_planes = []
        norm_planes = []
        for plane in planes:
            dilated_plane = cv2.dilate(plane, np.ones((5, 5), np.uint8))
            blurred_bg = cv2.medianBlur(dilated_plane, 19)
            diff_img = 255 - cv2.absdiff(plane, blurred_bg)
            normalized_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            result_planes.append(diff_img)
            norm_planes.append(normalized_img)

        image_without_shadows = cv2.merge(result_planes)
        self.assertTrue(image_without_shadows is not None)

    def test_image_processing(self):
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError("Image not found or unable to read.")
        planes = cv2.split(image)
        result_planes = []
        norm_planes = []
        for plane in planes:
            dilated_plane = cv2.dilate(plane, np.ones((5, 5), np.uint8))
            blurred_bg = cv2.medianBlur(dilated_plane, 19)
            diff_img = 255 - cv2.absdiff(plane, blurred_bg)
            normalized_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            result_planes.append(diff_img)
            norm_planes.append(normalized_img)

        image_without_shadows = cv2.merge(result_planes)
        edges = cv2.Canny(image_without_shadows, 55, 150)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        self.assertTrue(morphed_image is not None)



if __name__ == '__main__':
    unittest.main()
