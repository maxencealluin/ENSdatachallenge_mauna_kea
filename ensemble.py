from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import keras
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
import os

#preparing data_augmentation generator

seq = iaa.SomeOf((0,2), [iaa.GaussianBlur(sigma = (0, 3.0)), iaa.AdditiveGaussianNoise(scale = (0, 10))])
def imgaug(image):
	return seq.augment_image(image)

train_gen = ImageDataGenerator(
			rescale=1. / 255,
			vertical_flip = 1,
			horizontal_flip = 1,
			rotation_range = 90,
			brightness_range = [0.4, 1.2],
			# zca_whitening = True,
			shear_range = 7.5,
			width_shift_range = .1,
			height_shift_range = .1,
			fill_mode ='constant',
			preprocessing_function = imgaug)

valid_gen = ImageDataGenerator(rescale=1. / 255)

test_gen = ImageDataGenerator(rescale=1. / 255)

test_gen_tta = ImageDataGenerator(
			rescale=1. / 255,
			vertical_flip = 1,
			horizontal_flip = 1,
			rotation_range = 90)

# train/valid/test generators

def define_generators(input_size, batch_size = 64):
	mode = 'grayscale' if (input_size[-1] == 1) else 'rgb'
	training_generator = train_gen.flow_from_dataframe(
					df_train,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'class_number',
					target_size = input_size[:2],
					color_mode = mode,
					batch_size = batch_size,
					shuffle = True,
					class_mode = 'categorical')
	if df_val == None:
		return training_generator, None
	valid_generator = valid_gen.flow_from_dataframe(
					df_val,
					directory = './mldata/train_set',
					x_col = 'image_filename',
					y_col = 'class_number',
					target_size = input_size[:2],
					color_mode = mode,
					batch_size = batch_size,
					shuffle = True,
					class_mode = 'categorical')
	return training_generator, valid_generator

from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout, GlobalAveragePooling2D, Lambda, BatchNormalization
from keras.regularizers import l2

from keras.applications.vgg16 import VGG16
#from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.xception import Xception, preprocess_input
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from keras.applications.densenet import DenseNet121
from keras.optimizers import Adam, SGD
from autoencoder import Autoencoder

class Kmodel():
	def __init__(self, name = 'default', input_size = (224, 224, 3), lr = 0.001):
		self.name = name
		self.input_size = input_size
		self.lr = lr
	def load_weights(self):
		try:
			self.model.load_weights('./mldata/weights/' + str(self.name) + '/current_model.hdf5')
			print(str(self.name) + ' weights loaded from ./mldata/weights/' + str(self.name) + '/current_model.hdf5')
		except:
			print("\nError loading weights\n")
	def	save_weights(self):
		if not os.path.exists('./mldata/weights/' + str(self.name)):
			os.mkdir('./mldata/weights/' + str(self.name))
		try:
			self.model.save('./mldata/weights/' + str(self.name) + '/current_model.hdf5')
			print(str(self.name) + ' weights saved in ./mldata/weights/' + str(self.name) + '/current_model.hdf5')
		except:
			print("\nError saving weights\n")
	def	summary(self):
		self.model.summary()
	def	print_layers(self):
		for layer in self.model:
			print(layer)
	def get_generators(self, batch_size = 64):
		self.train_gen, self.val_gen = define_generators(self.input_size, batch_size)
	def print_config(self):
		for layer in self.model:
			print(layer.get_config())


#manual model
class custom_model(Kmodel):
	def __init__(self, name, lr=0.001):
			super().__init__(name = name,  lr = lr)
			self.input_size = (512,512,1)
			self.model = Sequential()
			autoencoder_model = Autoencoder(trainable = 0)
			# autoencoder_model.load_weights()
			self.model.add(autoencoder_model.encoder)
			self.model.add(Conv2D(filters = 128, kernel_size = (3,3), input_shape = (64,64,128), strides = 1, padding ='same', activation = 'relu'))
			self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

			self.model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
			self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

			self.model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
			self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

			# self.model.add(Conv2D(filters = 512, kernel_size = (3,3), strides = 1, padding ='same', activation = 'relu'))
			# self.model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

			self.model.add(GlobalAveragePooling2D())
			self.model.add(Dense(2048, activation='relu'))
			self.model.add(Dense(1024, activation='relu'))
			self.model.add(Dense(4, activation='softmax'))

			sgd_opti = SGD(lr = self.lr, momentum = 0.9)
			self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

# out of the box model training from scratch
class resnext_scratch(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)
		weight_decay = 0.0001
		res = ResNet50(include_top = False, weights = None, classes = 4)
		for layer in res.layers:
			if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
				layer.kernel_regularizer = l2(weight_decay)
			if hasattr(layer, 'bias_regularizer') and layer.use_bias:
				layer.bias_regularizer = l2(weight_decay)

		self.input_size = (448,448,1)
		self.model = Sequential()
		self.model.add(Conv2D(filters = 3, kernel_size = (3,3), input_shape = self.input_size, strides = 2, padding ='same', activation = 'relu'))
		self.model.add(res)
		self.model.add(GlobalAveragePooling2D())
		self.model.add(Dropout(rate = 0.5))
		self.model.add(Dense(2048, activation='relu'))
		self.model.add(Dropout(rate = 0.5))
		self.model.add(Dense(4, activation='softmax'))

		for layer in self.model.layers:
			if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
				layer.kernel_regularizer = l2(weight_decay)
			if hasattr(layer, 'bias_regularizer') and layer.use_bias:
				layer.bias_regularizer = l2(weight_decay)

		# adam_opti = Adam(lr = learning_rate)
		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

class pretrained_resnet(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)
		weight_decay = 0.0001
		pretrained = ResNet50V2(include_top = False, weights = 'imagenet', classes = 4)

		self.input_size = (224,224, 3)
		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(GlobalAveragePooling2D())
		self.model.add(Dropout(rate = 0.5))
		self.model.add(Dense(2048, activation='relu'))
		self.model.add(Dropout(rate = 0.5))
		self.model.add(Dense(4, activation='softmax'))

		# adam_opti = Adam(lr = learning_rate)
		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

# transfer learning pretrained model
class pretrained_vgg16(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)

		self.input_size = (224,224, 3)
		pretrained = VGG16(include_top =False, weights = 'imagenet', input_shape= self.input_size)
		for layer in pretrained.layers[:7]:
			layer.trainable = False

		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(Flatten())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(1024, activation='relu'))
		self.model.add(Dropout(0.25))
		self.model.add(Dense(512, activation='relu'))
		self.model.add(Dense(4, activation='softmax'))

		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

class pretrained_xception(Kmodel):
	def __init__(self, name, lr = 0.001):
		super().__init__(name = name,  lr = lr)
		self.input_size = (299,299, 3)
		weight_decay = 0.00005
		pretrained = Xception(include_top = False, weights = 'imagenet', input_shape = self.input_size, classes = 4)
		# for layer in pretrained.layers[:126]:
		# 	if not isinstance(layer, BatchNormalization):
		# 		layer.trainable = False
		for layer in pretrained.layers:
			if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
				layer.kernel_regularizer = l2(weight_decay)
			if hasattr(layer, 'bias_regularizer') and layer.use_bias:
				layer.bias_regularizer = l2(weight_decay)
		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(GlobalAveragePooling2D())
		# self.model.add(Dense(2048, activation='relu'))
		self.model.add(Dropout(0.5))
		# self.model.add(Dense(1024, activation='relu'))
		# self.model.add(Dropout(0.25))
		self.model.add(Dense(4, activation='softmax'))

		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

class pretrained_mobilenet(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)
		self.input_size = (224,224, 3)
		pretrained = MobileNetV2(include_top =False,  weights='imagenet', input_shape= self.input_size, classes = 4)
		# for layer in pretrained.layers[:-6]:
			# layer.trainable = False
		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(GlobalAveragePooling2D())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(4, activation='softmax'))

		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

class pretrained_nasnet(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)
		self.input_size = (224,224, 3)
		pretrained = NASNetMobile(input_shape=self.input_size, include_top=False, weights='imagenet', classes=4)
		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(GlobalAveragePooling2D())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(4, activation='softmax'))

		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

class pretrained_densenet(Kmodel):
	def __init__(self, name, lr=0.001):
		super().__init__(name = name,  lr = lr)
		self.input_size = (224,224, 3)
		pretrained = DenseNet121(input_shape=self.input_size, include_top=False, weights='imagenet', classes=4)
		self.model = Sequential()
		self.model.add(pretrained)
		self.model.add(GlobalAveragePooling2D())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(4, activation='softmax'))

		sgd_opti = SGD(lr = self.lr, momentum = 0.9)
		self.model.compile(optimizer = sgd_opti, loss='categorical_crossentropy', metrics = ['accuracy'])

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#callbacks

callbacks = []

#tensorboard to log training data
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

callbacks.append(TensorBoard(log_dir = './mldata/logs/{}'.format(time())))

# filepath="./mldata/weights/weights-improvement-{epoch:02d}-{train_acc:.3f}.hdf5"
# callbacks.append(ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 1))

# callbacks.append(EarlyStopping(monitor = 'val_acc', patience = 5, verbose = 0, min_delta=0.005))

#Schedule to use for SGD
def schedule(epoch, curr_lr):
	if curr_lr <= 0.0001:
		return curr_lr
	if epoch < 40:
		return 0.0150
	elif epoch < 70:
		return 0.0075
	elif epoch < 120:
		return 0.0012
	else:
		return 0.0008

def schedule_decrease(epoch, curr_lr):
	if epoch % 2 == 0 and epoch > 0:
		return curr_lr * 0.9
	return curr_lr

callbacks.append(LearningRateScheduler(schedule_decrease, verbose = 1))
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, min_lr=0.0002))

#model training
def train_model(training_model, nb_epochs, weights = None):
	print("Training " + str(training_model.name) + " ...")
	training_generator = training_model.train_gen
	valid_generator = training_model.val_gen
	train_model = training_model.model
	filepath="./mldata/weights/" + str(training_model.name) + "/weights-{epoch:02d}-{acc:.3f}.hdf5"
	train_callbacks = callbacks
	train_callbacks.append(ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max', period = 10))
	train_model.fit_generator(training_generator,
						epochs = nb_epochs,
						steps_per_epoch = training_generator.samples // training_generator.batch_size + 1,
						callbacks = train_callbacks,
						class_weight = weights)#,)
						# validation_data = valid_generator,
						# validation_steps = valid_generator.samples // valid_generator.batch_size + 1,

#cross val
def predict_val(val_model):
	print("Cross val " + str(val_model.name) + " ...")
	mode = 'grayscale' if (val_model.input_size[-1] == 1) else 'rgb'
	_, df_val_tmp = load_to_dataframe("./mldata/train_y.csv", shuffle = True)
	val_generator = test_gen.flow_from_dataframe(
				df_val_tmp,
				directory = './mldata/train_set/',
				x_col = 'image_filename',
				target_size = val_model.input_size[:2],
				color_mode = mode,
				shuffle = False,
				classes = [0, 1, 2, 3],
				batch_size = 1,
				class_mode = None)
	results = val_model.model.predict_generator(val_generator, steps = val_generator.samples)
	df_val_tmp['class_number'] = np.argmax(results, 1)
	df_val_tmp.to_csv('/home/seldon/mauna_kea/mldata/cv/results_val_' + str(val_model.name) + '.csv', index = None, header=True)

### predict test results
def predict_test(testing_model):
	print("Testing " + str(testing_model.name) + " ...")
	mode = 'grayscale' if (testing_model.input_size[-1] == 1) else 'rgb'
	df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	test_generator = test_gen.flow_from_dataframe(
					df_test,
					directory = './mldata/test_set/',
					x_col = 'image_filename',
					target_size = testing_model.input_size[:2],
					color_mode = mode,
					shuffle = False,
					classes = [0, 1, 2, 3],
					batch_size = 1,
					class_mode = None)
	test_model = testing_model.model
	results = test_model.predict_generator(test_generator, steps = test_generator.samples)
	df_test['class_number'] = np.argmax(results, 1)
	df_test.to_csv ('/home/seldon/mauna_kea/mldata/results_' + str(testing_model.name) + '.csv', index = None, header=True)

def predict_test_tta(testing_model, epochs = 5):
	print("Testing " + str(testing_model.name) + " ...")
	mode = 'grayscale' if (testing_model.input_size[-1] == 1) else 'rgb'
	df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	test_generator = test_gen_tta.flow_from_dataframe(
					df_test,
					directory = './mldata/test_set/',
					x_col = 'image_filename',
					target_size = testing_model.input_size[:2],
					color_mode = mode,
					shuffle = False,
					classes = [0, 1, 2, 3],
					batch_size = 1,
					class_mode = None)
	predictions = []
	test_model = testing_model.model
	for i in range(epochs):
		preds = test_model.predict_generator(test_generator, steps = test_generator.samples)
		predictions.append(preds)
	pred = np.mean(predictions, axis = 0)
	df_test['class_number'] = np.argmax(pred, 1)
	df_test.to_csv ('/home/seldon/mauna_kea/mldata/results_' + str(testing_model.name) + '.csv', index = None, header=True)

def predict_test_mult(gen, testing_model, df_test):
	print("Testing " + str(testing_model.name) + " ...")
	mode = 'grayscale' if (testing_model.input_size[-1] == 1) else 'rgb'
	# df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	test_generator = gen.flow_from_dataframe(
					df_test,
					directory = './mldata/test_set/',
					x_col = 'image_filename',
					target_size = testing_model.input_size[:2],
					color_mode = mode,
					shuffle = False,
					classes = [0, 1, 2, 3],
					batch_size = 1,
					class_mode = None)
	test_model = testing_model.model
	results = test_model.predict_generator(test_generator, steps = test_generator.samples)
	return results

def predict_test_mult_tta(gen, testing_model, df_test, epochs = 5):
	print("Testing tta " + str(testing_model.name) + " ...")
	mode = 'grayscale' if (testing_model.input_size[-1] == 1) else 'rgb'
	# df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	test_generator = gen.flow_from_dataframe(
					df_test,
					directory = './mldata/test_set/',
					x_col = 'image_filename',
					target_size = testing_model.input_size[:2],
					color_mode = mode,
					shuffle = False,
					classes = [0, 1, 2, 3],
					batch_size = 1,
					class_mode = None)
	predictions = []
	test_model = testing_model.model
	for i in range(epochs):
		print(i, end = '')
		preds = test_model.predict_generator(test_generator, steps = test_generator.samples)
		predictions.append(preds)
	pred = np.mean(predictions, axis = 0)
	return pred

if __name__ == '__main__':
	#LOADING DATA
	from load_data import load_to_dataframe
	# df_train, df_val, class_weights = load_to_dataframe("./mldata/train_y.csv", shuffle = True, weights = 1)
	df_train, class_weights = load_to_dataframe("./mldata/train_y.csv", split = 0, shuffle = True, weights = 1)
	df_val = None
	# Models
	models = []
	# models.append(custom_model(name = 'autoencoder', lr = 0.001))
	models.append(resnext_scratch(name = 'resnext', lr = 0.0008))
	models.append(pretrained_resnet(name = 'pretrained_resnetv2', lr = 0.001)) #start at 0.01
	models.append(pretrained_vgg16(name = 'pretrained_vgg', lr = 0.0005))
	models.append(pretrained_xception(name = 'xception', lr = 0.005)) # start at 0.045
	models.append(pretrained_mobilenet(name = 'mobilenet', lr = 0.01))
	models.append(pretrained_nasnet(name = 'nasnet', lr = 0.002)) # start at 0.01
	models.append(pretrained_densenet(name = 'densenet', lr = 0.0001)) # start at 0.001



	list_epochs = [10]#, 80, 80]
	results = []
	results_tta = []
	#TRAINING/LOADING
	for i, model in enumerate(models):
		print('\n' + str.upper(model.name))
		model.summary()
		if model.name == 'xception':
			model.get_generators(16)
		elif model.name == 'densenet':
			model.get_generators(32)
		else:
			model.get_generators(64)
		model.load_weights()
		# train_model(model, list_epochs[i], class_weights)
		# train_model(model, 1)
		# model.save_weights()
		# predict_val(model)
		# predict_test(model)
		df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
		results.append(predict_test_mult(test_gen, model, df_test))
		# results_tta.append(predict_test_mult_tta(test_gen_tta, model, df_test, 5))
	# exit()

	def pseudo_labelling(results):
		preds = np.mean(results, axis = 0)
		count = 0
		copy = {}
		for idx, pred in enumerate(preds):
			max_val =  max(pred)
			if max_val > 0.975:
				copy[idx] = np.argmax(pred, 0)
				count += 1
		print(count)
		cop = {2:3, 4:1, 6:2, 12:0}
		keys = list(copy.keys())
		vals = list(copy.values())
		df_conv = pd.DataFrame()
		df_test = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
		df_conv['image_filename'] = df_test.iloc[keys]['image_filename']
		df_conv['class_number'] = vals
		df_conv.to_csv('/home/seldon/mauna_kea/mldata/train_y_add.csv', index = None, header=True)

	pseudo_labelling(results)
	exit()

	df_test_final = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	df_test_final['class_number'] = np.argmax(preds, 1)
	df_test_final.to_csv ('/home/seldon/mauna_kea/mldata/results_ensemble.csv', index = None, header=True)
	print("Saved to results_ensemble.csv")
	preds_tta = np.mean(results_tta, axis = 0)
	df_test_final_tta = load_to_dataframe("./mldata/test_data_order.csv", y_label= 0, split = 0, shuffle = 0)
	df_test_final_tta['class_number'] = np.argmax(preds_tta, 1)
	df_test_final_tta.to_csv ('/home/seldon/mauna_kea/mldata/results_ensemble_tta.csv', index = None, header=True)
	print("Saved to results_ensemble_tta.csv")
