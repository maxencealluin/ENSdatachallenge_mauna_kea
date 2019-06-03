import numpy as np
import pandas as pd
from PIL import Image

def load_img(path):
	data = np.array(Image.open(path).convert("L"))
	return data

def load_to_dataframe(path_y, path_dir, y_label = 1):
	df = pd.read_csv(path_y)
	df['image_filename'] = '/Users/malluin/goinfre/train/' + df['image_filename']
	df['image_filename'] = df['image_filename'].apply(load_img)
	return df

df = load_to_dataframe("/Users/malluin/goinfre/test_y.csv", "/Users/malluin/goinfre/train")
print(df)
