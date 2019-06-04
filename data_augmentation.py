from PIL import Image
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob
import matplotlib.pyplot as plt
import random
import imgaug.augmenters as iaa

def load_images_np(path):
	dataset =[]
	i = 0
	for file in glob.glob(path):
		dataset.append(np.array(Image.open(file).convert("L")))
		i += 1
		if i == 18:
			break
	dataset = np.array(dataset)
	dataset = np.expand_dims(dataset, axis = 3)
	return (dataset)

def plot_image_np(data):
	fig, axes = plt.subplots(3, 6, sharex='col', sharey='row', squeeze=True)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
	# rands = np.squeeze(data[np.random.choice(data.shape[0], 18)])
	rands = np.squeeze(data)
	for i, ax in enumerate(axes.flat):
		img = Image.fromarray(rands[i])
		ax.imshow(img)
		ax.axis("off")
	plt.show()
#
test = load_images_np("/Users/malluin/goinfre/testset/**/**")
# plot_image_np(test)
# data_gen = ImageDataGenerator()

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

train_gen.fit(test)

for batch in train_gen.flow(test):
	print(batch.shape)
	plot_image_np(batch)
