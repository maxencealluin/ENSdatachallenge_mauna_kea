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
			rescale=1. / 255,
			vertical_flip = 1,
			horizontal_flip = 1,
			rotation_range = 360,
			brightness_range = [0.3, 1.5],
			shear_range = 10,
			width_shift_range = .1,
			height_shift_range = .1,
			fill_mode ='constant',
			preprocessing_function = imgaug)

valid_gen = ImageDataGenerator(rescale=1. / 255)

test_gen = ImageDataGenerator(rescale=1. / 255)

#loading data

from load_data import load_to_dataframe

df_train, df_val =  load_to_dataframe("./mldata/train_y.csv", shuffle = True)

input_size = (256, 256)

# train/valid/test generators

def define_generators(input_size, mode = 'grayscale'):
	global training_generator
	global valid_generator
	training_generator = train_gen.flow_from_dataframe(
					df_train,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'class_number',
					target_size = input_size,
					color_mode = mode,
					batch_size = 64,
					shuffle = True,
					class_mode = 'categorical')

	valid_generator = valid_gen.flow_from_dataframe(
					df_val,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'class_number',
					target_size = input_size,
					color_mode = mode,
					batch_size = 64,
					shuffle = True,
					class_mode = 'categorical')

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2

from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception

#manual model
def custom_model():
	model = Sequential()
	model.add(Conv2D(filters = 128, kernel_size = (3,3), input_shape = (256,256,1), strides = 2, padding ='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

	model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 2, padding ='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

	model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

	model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
	model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

	# model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
	# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

	# model.add(Flatten())
	model.add(GlobalAveragePooling2D())
	model.add(Dense(2048, activation='relu'))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(4, activation='softmax'))

	model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
	model.summary()
	return model


# out of the box model training from scratch
def vgg16_scratch(learning_rate = 0.0001):
	# vgg = VGG19(weights = None, classes = 4)
	vgg = VGG16(weights = None, classes = 4)

	model = Sequential()
	model.add(Conv2D(filters = 3, kernel_size = (3,3), input_shape = (224,224,1), strides = 1, padding ='same', activation = 'relu'))

	for layer in vgg.layers:
		# print(layer.name)
		if "block" in layer.name:
			model.add(layer)

	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(rate = 0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(rate = 0.5))
	model.add(Dense(4, activation='softmax'))

	from keras.optimizers import Adam
	adam_opti = Adam(lr = learning_rate)
	model.compile(optimizer = adam_opti, loss='categorical_crossentropy', metrics = ['accuracy'])
	model.summary()
	return model

# out of the box model training from scratch
def resnext_scratch(learning_rate = 0.1):

	weight_decay = 0.0001
	res = ResNet50(include_top = False, weights = None, classes = 4)
	# res.summary()
	for layer in res.layers:
		if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
			layer.kernel_regularizer = l2(weight_decay)
		if hasattr(layer, 'bias_regularizer') and layer.use_bias:
			layer.bias_regularizer = l2(weight_decay)

	model = Sequential()
	model.add(Conv2D(filters = 3, kernel_size = (3,3), input_shape = (224,224,1), strides = 1, padding ='same', activation = 'relu'))


	model.add(res)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(rate = 0.5))
	model.add(Dense(4, activation='softmax'))

	for layer in model.layers:
		if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
			layer.kernel_regularizer = l2(weight_decay)
		if hasattr(layer, 'bias_regularizer') and layer.use_bias:
			layer.bias_regularizer = l2(weight_decay)

	# for layer in model.layers:
	# 	print(layer.get_config())
	# 	print("\n")

	from keras.optimizers import Adam, SGD
	# adam_opti = Adam(lr = learning_rate)
	sgd_opti = SGD(lr = learning_rate, momentum = 0.9)
	model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])
	model.summary()
	return model

# transfer learning pretrained model

def transfer_learning(learning_rate = 0.0001):

	pretrained = VGG16(include_top =False, weights = 'imagenet', input_shape= (224,224, 3))
	# pretrained = Xception(include_top =False,  weights='imagenet', input_shape= (299,299, 3), classes = 4)

	for layer in pretrained.layers:
		layer.trainable = False

	model = Sequential()
	model.add(pretrained)

	model.add(Flatten())
	# model.add(Dropout(0.5))
	model.add(Dense(2048, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(4, activation='softmax'))

	from keras.optimizers import Adam
	adam_opti = Adam(lr = learning_rate)
	model.compile(optimizer = adam_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

	# pretrained.summary()
	model.summary()

	# for layer in model.layers:
	# 	print(layer)
	return model


# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#callbacks

callbacks = []

#tensorboard to log training data
from time import time
from tensorflow.python.keras.callbacks   import TensorBoard

callbacks.append(TensorBoard(log_dir = './mldata/logs/{}'.format(time())))

from keras.callbacks import ModelCheckpoint
filepath="./mldata/weights/weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
callbacks.append(ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 10))

from keras.callbacks import EarlyStopping
# callbacks.append(EarlyStopping(monitor = 'val_acc', patience = 5, verbose = 0, min_delta=0.005))

#Schedule to use for SGD
from keras.callbacks import LearningRateScheduler

def schedule(epoch, curr_lr):
	if curr_lr <= 0.0001:
		return curr_lr
	if epoch < 80:
		return 0.01
	elif epoch < 120:
		return 0.001
	else:
		return 0.0001

#callbacks.append(LearningRateScheduler(schedule, verbose = 1))

#model trainingwith

def load_weights(file):
	try:
		model.load_weights('./mldata/weights/' + str(file))
		print("\nWeights loaded from ./mldata/weights/" + str(file) + "\n")
	except:
		print("\nError loading weights\n")

def train_model(nb_epochs):
	model.fit_generator(training_generator,
						epochs = nb_epochs,
						steps_per_epoch = training_generator.samples // training_generator.batch_size + 1,
						callbacks = callbacks,
						validation_data = valid_generator,
						validation_steps = valid_generator.samples // valid_generator.batch_size + 1)
	model.save('./mldata/weights/current_model')

### predict test results
def predict_test(input_size, mode = 'grayscale'):
	df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	test_generator = test_gen.flow_from_dataframe(
					df_test,
					directory = './mldata/test_set/',
					x_col = 'image_filename',
					target_size = input_size,
					color_mode = mode,
					shuffle = False,
					classes = [0, 1, 2, 3],
					batch_size = 1,
					class_mode = None)
	results = model.predict_generator(test_generator, steps = test_generator.samples)
	df_test['class_number'] = np.argmax(results, 1)
	df_test.to_csv ('/home/seldon/mauna_kea/mldata/results.csv', index = None, header=True)

# def predict_verif(input_size, mode = 'grayscale'):
# 	df_test = load_to_dataframe("./mldata/train_y.csv", y_label= 0, split = 0, shuffle = 0)
# 	test_generator = test_gen.flow_from_dataframe(
# 					df_test,
# 					directory = './mldata/train_set/',
# 					x_col = 'image_filename',
# 					target_size = input_size,
# 					color_mode = mode,
# 					batch_size = 1,
# 					shuffle = False,
# 					classes = [0, 1, 2, 3],
# 					class_mode = None)
# 	# print(test_generator.class_indices)
# 	results = model.predict_generator(test_generator, steps = test_generator.samples)
# 	df_test['class_number'] = np.argmax(results, 1)
# 	df_test.to_csv('/home/seldon/mauna_kea/mldata/try.csv', index = None, header=True)

cnn_input_size = (224,224)

# Chosoe good input according to model
define_generators(cnn_input_size, 'grayscale')
# Choose one Model
# model = custom_model()
# model = vgg16_scratch(0.00005)
model = resnext_scratch(0.0007)
# model = transfer_learning(0.0001)

#TRAINING
load_weights("current_model")
train_model(50)
# predict_verif(cnn_input_size)
predict_test(cnn_input_size)
