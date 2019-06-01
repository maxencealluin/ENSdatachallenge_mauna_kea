from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

X_train = []
for file in glob.glob("/Users/malluin/goinfre/testset/**/**"):
	X_train.append(Image.open(file))
fig, axes = plt.subplots(3, 6, sharex='col', sharey='row', squeeze=True)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.tight_layout(pad=0, w_pad=0, h_pad=0.0)
rands = random.sample(X_train, 18)
for i, ax in enumerate(axes.flat):
	ax.imshow(rands[i])
	ax.axis("off")
plt.show()
