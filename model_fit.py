from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

from data_augmentation.py import load_images_np

seq = iaa.SomeOf((0,2), [iaa.GaussianBlur(sigma = (0,5)), iaa.AdditiveGaussianNoise(scale = (0,15))])
def imgaug(image):
	return seq.augment_image(image)

train_gen = ImageDataGenerator(
			vertical_flip = 1,
			horizontal_flip = 1,
			rotation_range = 90,
			brightness_range = [0.3, 1.5],
			shear_range = 10,
			width_shift_range = .1,
			height_shift_range = .1,
			fill_mode ='constant',
			preprocessing_function = imgaug)

test_gen = ImageDataGenerator()

training_generator = train_gen.flow_from_dataframe(
				"/Users/malluin/goinfre/train/",
				batch_size = 64,
				class_mode = 'categorical')

valid_generator = test_gen.flow_from_dataframe(
				"/Users/malluin/goinfre/valid/",
				batch_size = 64,
				class_mode = 'categorical')

test_generator = test_gen.flow_from_dataframe(
				"/Users/malluin/goinfre/test/",
				batch_size = 64,
				class_mode = None)


##model

model.fit_generator(train_generator,
bath_size = 64,
epochs = 50,
validation_data = valid_generator)
