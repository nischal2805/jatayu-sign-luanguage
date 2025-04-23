def load_image(image_path):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path)
    return np.array(image)

def save_image(image_array, save_path):
    from PIL import Image

    image = Image.fromarray(image_array)
    image.save(save_path)

def resize_image(image_array, target_size):
    from PIL import Image

    image = Image.fromarray(image_array)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)

def normalize_image(image_array):
    return image_array / 255.0

def convert_to_grayscale(image_array):
    from PIL import Image

    image = Image.fromarray(image_array)
    grayscale_image = image.convert('L')
    return np.array(grayscale_image)