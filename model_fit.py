from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa

#preparing data_augmentation generator

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

#loading data

# from load_data import load_to_dataframe

# df_train = load_to_dataframe("/Users/malluin/goinfre/train_y.csv", "/Users/malluin/goinfre/train")

# train/valid/test generators

# training_generator = train_gen.flow_from_dataframe(
# 				df_train,
# 				directory = '/Users/malluin/goinfre/train/',
# 				x_col = 'image_filename',
# 				y_col = 'class_number',
# 				target_size = (520,520),
# 				color_mode = 'grayscale',
# 				batch_size = 64,
# 				class_mode = 'categorical')

# valid_generator = test_gen.flow_from_dataframe(
# 				df_train,
# 				directory = '/Users/malluin/goinfre/train/',
# 				x_col = 'image_filename',
# 				y_col = 'class_number',
# 				target_size = (520,520),
# 				color_mode = 'grayscale',
# 				batch_size = 64,
# 				class_mode = 'categorical')

# test_generator = test_gen.flow_from_dataframe(
# 				"/Users/malluin/goinfre/test/",
# 				batch_size = 64,
# 				class_mode = None)

#mockup of model

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout

# model = Sequential()
# model.add(Conv2D(filters = 64, kernel_size = (5,5), input_shape = (520,520,1), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 64, kernel_size = (5,5), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

# model.add(Flatten())
# model.add(Dense(4225, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(4, activation='softmax'))

# model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])


# transfer learning vgg 16

from keras.applications.vgg16 import VGG16
vgg = VGG16(include_top =False, input_shape= (260,260, 3))

for layer in vgg.layers:
	layer.trainable = False

model = Sequential()
model.add(vgg)

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

for layer in model.layers:
	print(layer)

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#model training

model.fit_generator(training_generator,
					epochs = 50,
					steps_per_epoch = 1000,
					validation_data = valid_generator,
					validation_steps = 200)
