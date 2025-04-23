import os
import cv2
import numpy as np

class DataPreprocessor:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at path: {image_path}")
        return image

    def resize_image(self, image):
        return cv2.resize(image, self.image_size)

    def normalize_image(self, image):
        return image / 255.0

    def preprocess_image(self, image_path):
        image = self.load_image(image_path)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image

    def preprocess_dataset(self, dataset_path):
        processed_images = []
        for filename in os.listdir(dataset_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dataset_path, filename)
                processed_image = self.preprocess_image(image_path)
                processed_images.append(processed_image)
        return np.array(processed_images)