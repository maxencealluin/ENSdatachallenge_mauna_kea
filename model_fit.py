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
			brightness_range = [0.3, 1],
			shear_range = 10,
			width_shift_range = .1,
			height_shift_range = .1,
			fill_mode ='constant',
            validation_split=0.2,
			preprocessing_function = imgaug)

valid_gen = ImageDataGenerator(rescale=1. / 255)

test_gen = ImageDataGenerator(rescale=1. / 255)

#loading data

from load_data import load_to_dataframe

df_train, df_val =  load_to_dataframe("./mldata/train_y.csv", "./mldata/train")



input_size = (224, 224)

# train/valid/test generators

training_generator = train_gen.flow_from_dataframe(
				df_train,
				directory = './mldata/train_set',
				x_col = 'image_filename',
				y_col = 'class_number',
				target_size = input_size,
				color_mode = 'grayscale',
				batch_size = 64,
                shuffle = True,
				class_mode = 'categorical')
                # subset = 'training')
                
valid_generator = valid_gen.flow_from_dataframe(
				df_val,
				directory = './mldata/train_set',
				x_col = 'image_filename',
				y_col = 'class_number',
				target_size = input_size,
				color_mode = 'grayscale',
				batch_size = 64,
                shuffle = True,
				class_mode = 'categorical')
                # subset = 'validation')


from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, ZeroPadding2D, Dropout, GlobalAveragePooling2D
from keras import regularizers

# from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19


'''
#manul model

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
'''


# out of the box model trianing from scratch

vgg = VGG19(weights = None, classes = 4)

model = Sequential()
model.add(Conv2D(filters = 3, kernel_size = (3,3), input_shape = (224,224,1), strides = 1, padding ='same', activation = 'relu'))

for layer in vgg.layers:
    if "block" in layer.name:
        model.add(layer)

model.add(GlobalAveragePooling2D())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(4, activation='softmax'))

# from keras.optimizers import Adam
# adam = Adam(lr = 0.01)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()


'''
# transfer learning pretrained model


from keras.applications.xception import Xception

vgg = VGG19(include_top =False, input_shape= (224,224, 3))
# pretrained = Xception(include_top =False,  weights='imagenet', input_shape= (299,299, 3), classes = 4)

for layer in pretrained.layers:
	layer.trainable = False

model = Sequential()
model.add(pretrained)

model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu', kernel_regularizer = regularizers.l1(0.001)))
# model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer = regularizers.l1(0.001)))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])

# pretrained.summary()
# model.summary()

# for layer in model.layers:
# 	print(layer)

'''

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#callbacks

callbacks = []

#tensorboard to log training data
from time import time
from tensorflow.python.keras.callbacks   import TensorBoard

callbacks.append(TensorBoard(log_dir = './mldata/logs/{}'.format(time())))

from keras.callbacks import ModelCheckpoint
filepath="./mldata/weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
callbacks.append(ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max', period = 10))

from keras.callbacks import EarlyStopping

# callbacks.append(EarlyStopping(monitor = 'val_acc', patience = 5, verbose = 0, min_delta=0.005))

#model training

try:
    model.load_weights('./mldata/weights/current_model')
except:
    print("\nError loading weights\n")

model.fit_generator(training_generator,
					epochs = 500,
					steps_per_epoch = training_generator.samples // training_generator.batch_size + 1,
                    callbacks = callbacks,
					validation_data = valid_generator,
					validation_steps = valid_generator.samples // valid_generator.batch_size + 1)

model.save('./mldata/weights/current_model')

### predict test results

df_test = load_to_dataframe("./mldata/test_data_order.csv", "./mldata/test", y_label= 0, split = 0)
test_generator = test_gen.flow_from_dataframe(
                df_test,
				directory = './mldata/test_set/',
                x_col = 'image_filename',
                target_size = input_size,
				color_mode = 'grayscale',
				batch_size = 1,
				class_mode = None)

results = model.predict_generator(test_generator, steps = test_generator.samples)
df_test['class_number'] = np.argmax(results, 1)
df_test.to_csv ('/home/seldon/mauna_kea/mldata/results.csv', index = None, header=True)

