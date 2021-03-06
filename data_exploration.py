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
	test = [x / data.shape[0] for x in classes.values()]
	for key, x in zip(classes.keys(), test):
		print("{} {:.3f}% ".format(key, x), end ='')
	print("\n")
	plt.show()

def load_images(path):
	dataset =[]
	i = 0
	for file in glob.glob(path):
		dataset.append(Image.open(file))
		i += 1
		if i > 100:
			break;
	print(dataset[0].size)
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

if __name__ == "__main__":
	X_test = load_images("./mldata/train_set/*")
	plot_sample(X_test)
# plot_classes(Y_train, 'class_number')
	Y_train = pd.read_csv("./mldata/train_y.csv")
	plot_classes(Y_train, 'class_number')
