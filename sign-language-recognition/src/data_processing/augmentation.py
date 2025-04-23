from keras.preprocessing.image import ImageDataGenerator

def create_data_augmentation_generator(rotation_range=20, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2,
                                       zoom_range=0.2, horizontal_flip=True,
                                       fill_mode='nearest'):
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode
    )
    return datagen

def augment_images(datagen, images, batch_size=32):
    return datagen.flow(images, batch_size=batch_size)