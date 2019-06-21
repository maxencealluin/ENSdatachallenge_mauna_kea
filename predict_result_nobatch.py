import keras
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
import pandas as pd
from PIL import Image
import numpy as np

def resnext_scratch(learning_rate = 0.1):
	weight_decay = 0.0001
	res = ResNet50(include_top = False, weights = None, classes = 4)
	# res.summary()
	for layer in res.layers:
		if isinstance(layer, Conv2D) or isinstance(layer, Dense):
			layer.kernel_regularizer = l2(weight_decay)
		if hasattr(layer, 'bias_regularizer') and layer.use_bias:
			layer.bias_regularizer = l2(weight_decay)

	model = Sequential()
	model.add(Conv2D(filters = 3, kernel_size = (3,3), input_shape = (448,448,1), strides = 2, padding ='same', activation = 'relu'))

	model.add(res)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(2048, activation='relu'))
	model.add(Dropout(rate = 0.5))
	model.add(Dense(4, activation='softmax'))

	for layer in model.layers:
		if isinstance(layer, Conv2D) or isinstance(layer, Dense):
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

def load_weights(file):
	try:
		model.load_weights('./mldata/weights/' + str(file))
		print("\nWeights loaded from ./mldata/weights/" + str(file) + "\n")
	except:
		print("\nError loading weights\n")

def load_img(path):
	data = np.array(Image.open(path).convert("L").resize((512, 512)))
	data = data / 255.0
	return data

def load_imgs_to_dataframe(path_y, path_folder):
	df = pd.read_csv(path_y)
	df['image_filename'] = path_folder + df['image_filename']
	df['image'] = df['image_filename'].apply(load_img)
	return df

if __name__ == '__main__':
	model = resnext_scratch()
	load_weights("current_model")
	df_test = load_imgs_to_dataframe("./mldata/test_data_order.csv", "./mldata/test_set/")
	X_test = np.ndarray(shape = (len(df_test['image']), 448, 448, 1))
	for i, img in enumerate(df_test['image']):
		X_test[i] = np.expand_dims(img, axis = 2)
	# X_test = df_test['image'].values
	results = model.predict(X_test)
	df_res = pd.DataFrame()
	df_res['image_filename'] = df_test['image_filename']
	df_res['class_number'] = np.argmax(results, 1)
	df_res.to_csv ('/home/seldon/mauna_kea/mldata/results_predict.csv', index = None, header=True)