from PIL import Image
import pandas as pd
import numpy as np
import glob
import imgaug.augmenters as iaa
from sys import argv

df_pred = pd.read_csv(argv[1])
print(df_pred.shape)

from data_exploration import plot_classes
plot_classes(df_pred, 'class_number')