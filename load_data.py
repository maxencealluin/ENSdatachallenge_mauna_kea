import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import class_weight

def load_img(path):
	data = Image.open(path).convert("L")
	data = np.array(data.resize((520,520)))
	return data

def load_to_dataframe(path_y, y_label = 1, split = 1, shuffle = 1, weights = 0):
	df = pd.read_csv(path_y)
	if y_label == 1:
		df['class_number'] = df['class_number'].astype(str)
	if shuffle == 1:
		df = df.sample(frac=1, random_state=42).reset_index(drop=True)
	if weights == 1:
		class_weights = class_weight.compute_class_weight('balanced', np.unique(df['class_number']), df['class_number'])
		class_weights = {k:v for (k,v) in zip([0,1,2,3], class_weights)}
	if split == 0:
		if weights == 1:
			return df, class_weights
		else:
			return df
	# msk = np.random.rand(len(df)) < 0.8
	# df_train = df[msk]
	# df_val = df[~msk]
	split_idx = int(df.shape[0] * 0.05)
	df_train = df.iloc[split_idx:,:]
	df_val = df.iloc[:split_idx,:]
	if weights == 1:
		return df_train, df_val, class_weights
	else:
		return df_train, df_val

if __name__ == "__main__":
	df_train, df_val = load_to_dataframe("./mldata/train_y.csv")
	print(df_train, df_val)

	from data_exploration import plot_classes
	plot_classes(df_train, 'class_number')
	plot_classes(df_val, 'class_number')


	df = load_to_dataframe("./mldata/train_y.csv", split = 0)
	# print(df)
