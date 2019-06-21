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
			# vertical_flip = 1,
			# horizontal_flip = 1,
			# rotation_range = 360,
			# brightness_range = [0.3, 1.5],
			# zca_whitening = True,
			# shear_range = 10,
			# width_shift_range = .1,
			# height_shift_range = .1,
			# fill_mode ='constant',
			preprocessing_function = imgaug)

valid_gen = ImageDataGenerator(rescale=1. / 255)


# train/valid/test generators

def define_generators(input_size, mode = 'grayscale'):
	global training_generator
	global valid_generator
	training_generator = train_gen.flow_from_dataframe(
					df_train,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'image_filename',
					target_size = input_size,
					color_mode = mode,
					batch_size = 16,
					class_mode = 'input',
					shuffle = True)

	valid_generator = valid_gen.flow_from_dataframe(
					df_val,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'image_filename',
					target_size = input_size,
					color_mode = mode,
					batch_size = 16,
					shuffle = True,
					class_mode = 'input')

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout, GlobalAveragePooling2D, UpSampling2D
from keras.regularizers import l2

from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
# from keras.applications.xception import Xception

#manual model
class Autoencoder:
	def __init__(self, trainable = 1):
		self.model = Sequential()
		self.encoder = Sequential()
		self.decoder = Sequential()
		self.encoder.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (512,512,1), strides = 2, padding ='same', activation = 'relu'))
		# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		self.encoder.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 2, padding ='same', activation = 'relu'))
		# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		self.encoder.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 2, padding ='same', activation = 'relu'))
		# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
		# model.add(Conv2D(filters = 16, kernel_size = (3,3), strides = 2, padding ='same', activation = 'relu'))
		# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
	
		# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		# model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

		# model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		# model.add(UpSampling2D((2,2)))
		# model.add(Conv2D(filters = 16, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		# model.add(UpSampling2D((2,2)))
		self.decoder.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		self.decoder.add(UpSampling2D((2,2)))
		self.decoder.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		self.decoder.add(UpSampling2D((2,2)))
		self.decoder.add(Conv2D(filters = 32, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
		self.decoder.add(UpSampling2D((2,2)))
		self.decoder.add(Conv2D(filters = 1, kernel_size = (3,3), strides = 1, padding ='same', activation = 'sigmoid'))
		self.model.add(self.encoder)
		self.model.add(self.decoder)

		from keras.optimizers import Adam, SGD
		adam_opti = Adam(lr = 0.0001)
		# sgd_opti = SGD(lr = learning_rate, momentum = 0.9)
		self.model.compile(optimizer = 'adadelta', loss='binary_crossentropy', metrics = ['accuracy'])
		if trainable == 0:
			for layer in self.model.layers:
				layer.trainable = 0
		# self.model.summary()
		# self.encoder.summary()
		# self.decoder.summary()

	def load_weights(self):
		try:
			self.encoder.load_weights('./mldata/weights/autoencoder/encoder.hdf5')
			self.decoder.load_weights('./mldata/weights/autoencoder/decoder.hdf5')
			print("\nEncoder weights loaded from ./mldata/weights/autoencoder/\n")
		except:
			print("\nError loading weights\n")
	def	save_weights(self):
		try:
			self.encoder.save('./mldata/weights/autoencoder/encoder.hdf5')
			self.decoder.save('./mldata/weights/autoencoder/decoder.hdf5')
			self.model.save('./mldata/weights/autoencoder/current_model.hdf5')
			print("\nEncoder weights saved inu ./mldata/weights/autoencoder/\n")
		except:
			print("\nError loading weights\n")

		
	def return_models(self):
		return self.model, self.encoder, self.decoder

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

'''

#callbacks

callbacks = []

#tensorboard to log training data
from time import time
from tensorflow.python.keras.callbacks   import TensorBoard

callbacks.append(TensorBoard(log_dir = './mldata/logs/{}'.format(time())))

from keras.callbacks import ModelCheckpoint
# filepath="./mldata/weights/weights-improvement-{epoch:02d}-{train_acc:.3f}.hdf5"
# callbacks.append(ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 1))

filepath="./mldata/weights/weights-improvement-{epoch:02d}-{acc:.3f}.hdf5"
callbacks.append(ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max', period = 10))

from keras.callbacks import EarlyStopping
# callbacks.append(EarlyStopping(monitor = 'val_acc', patience = 5, verbose = 0, min_delta=0.005))

#Schedule to use for SGD
from keras.callbacks import LearningRateScheduler

def schedule(epoch, curr_lr):
	if curr_lr <= 0.0001:
		return curr_lr
	if epoch < 40:
		return 0.015
	elif epoch < 70:
		return 0.01
	elif epoch < 120:
		return 0.0012
	else:
		return 0.0008

# callbacks.append(LearningRateScheduler(schedule, verbose = 1))'''

#model training

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
						# callbacks = callbacks),
						validation_data = valid_generator,
						validation_steps = valid_generator.samples // valid_generator.batch_size + 1)


# MAIN

if __name__ == '__main__':
	#loading data
	from load_data import load_to_dataframe

	df_train, df_val =  load_to_dataframe("./mldata/train_y.csv", shuffle = True)
	# df_train =  load_to_dataframe("./mldata/train_y.csv", shuffle = True, split = False)


	# Chosoe good input according to model
	cnn_input_size = (512,512)
	define_generators(cnn_input_size, 'grayscale')

	# Choose one Model
	# model = custom_model()
	# model = vgg16_scratch(0.00005)
	autoencoder_model = Autoencoder()
	autoencoder_model.load_weights()
	model = autoencoder_model.model
	# model = transfer_learning(0.0001)

	#TRAINING/LOADING
	train_model(5)
	autoencoder_model.save_weights()

	# predict_verif(cnn_input_size)

	def load_img(path):
		data = np.array(Image.open(path).convert("L").resize((512, 512)))
		data = data / 255.0
		return data

	def load_imgs_to_dataframe(path_y, path_folder):
		df = pd.read_csv(path_y)
		df = df.sample(n = 20)
		df['image_filename'] = path_folder + df['image_filename']
		df['image'] = df['image_filename'].apply(load_img)
		return df

	df_test =  load_imgs_to_dataframe("./mldata/test_data_order.csv", './mldata/test_set/')
	X_test = np.ndarray(shape = (20, 512, 512, 1))
	for i, img in enumerate(df_test['image']):
		X_test[i] = np.expand_dims(img, axis = 2)

	results = model.predict(X_test)
	import random
	def plot_sample(input, results):
		print(results.shape, input.shape)
		fig, axes = plt.subplots(4, 5, sharex='col', sharey='row', squeeze=True)
		plt.subplots_adjust(wspace=0.1, hspace=0.1)
		plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
		# rands = random.sample(data, 18)
		for i, ax in enumerate(axes.flat):
			if i % 10 < 5:
				ax.imshow(results[i].squeeze(), cmap='gray', vmin=0, vmax=255)
				ax.axis("off")
			else:
				ax.imshow(input[i- 5].squeeze(), cmap='gray', vmin=0, vmax=255)
				ax.axis("off")
		plt.show()

	results = results * 255.
	X_test = X_test * 255.
	plot_sample(X_test, results)