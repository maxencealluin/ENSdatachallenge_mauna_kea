from PIL import Image
import pandas as pd
import numpy as np
import glob
import imgaug.augmenters as iaa
from sys import argv

df_true = pd.read_csv(argv[1])
df_pred = pd.read_csv(argv[2])
print(df_true.shape)
print(df_pred.shape)

from sklearn.metrics import confusion_matrix, accuracy_score
matrix = confusion_matrix(df_true['class_number'], df_pred['class_number'])
accuracy = accuracy_score(df_true['class_number'], df_pred['class_number'])

print("Accuracy : {}".format(accuracy))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(matrix, annot=True,  fmt="d", vmin=0, vmax=20)
plt.show()