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

res = df_pred.merge(df_true, on=['image_filename'])
    
print(res)
from sklearn.metrics import confusion_matrix, accuracy_score
matrix = confusion_matrix(res['class_number_x'], res['class_number_y'])
accuracy = accuracy_score(res['class_number_x'], res['class_number_y'])

print("Accuracy : {}".format(accuracy))
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(matrix, annot=True,  fmt="d", vmin=0, vmax=20)
plt.show()