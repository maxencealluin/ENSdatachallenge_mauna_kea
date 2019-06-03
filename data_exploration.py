from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def plot_classes(data, col):
	classes = {}
	for ex in data[col]:
		name = 'class_' + str(ex)
		if (name in classes):
			classes[name] += 1
		else:
			classes[name] = 1
	x, y = list(classes.keys()), list(classes.values())
	print(classes)
	plt.bar(range(len(y)), y)
	plt.xticks(range(len(x)), x)
	plt.show()

def load_images(path):
	dataset =[]
	for file in glob.glob(path):
		dataset.append(Image.open(file))
	return (dataset)

def plot_sample(data):
	fig, axes = plt.subplots(3, 6, sharex='col', sharey='row', squeeze=True)
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
	rands = random.sample(data, 18)
	for i, ax in enumerate(axes.flat):
		ax.imshow(rands[i])
		ax.axis("off")
	plt.show()

X_test = load_images("/Users/malluin/goinfre/testset/**/**")
# plot_sample(X_test)

Y_train = pd.read_csv("/Users/malluin/goinfre/train_y.csv")
# plot_classes(Y_train, 'class_number')
