import unittest
from src.data_processing.preprocessor import preprocess_image
from src.data_processing.augmentation import augment_image

class TestDataProcessing(unittest.TestCase):

    def test_preprocess_image(self):
        # Test preprocessing of an image
        input_image = "path/to/raw/image.jpg"
        processed_image = preprocess_image(input_image)
        self.assertIsNotNone(processed_image)
        self.assertEqual(processed_image.shape, (224, 224, 3))  # Assuming the output shape is 224x224x3

    def test_augment_image(self):
        # Test augmentation of an image
        input_image = "path/to/raw/image.jpg"
        augmented_image = augment_image(input_image)
        self.assertIsNotNone(augmented_image)
        self.assertNotEqual(input_image, augmented_image)  # Ensure the augmented image is different

if __name__ == '__main__':
    unittest.main()