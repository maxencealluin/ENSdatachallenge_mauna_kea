import numpy as np
import pandas as pd
from PIL import Image

def load_img(path):
	data = Image.open(path).convert("L")
	data = np.array(data.resize((520,520)))
	return data

def load_to_dataframe(path_y, path_dir, y_label = 1, split = 1):
    df = pd.read_csv(path_y)
    if y_label == 1:
        df['class_number'] = df['class_number'].astype(str)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(df.shape[0] * 0.8)
    if split == 0:
        return df
    df_train = df.iloc[:split_idx,:]
    df_val = df.iloc[split_idx:,:]
    return df_train, df_val

# df_train, df_val = load_to_dataframe("./mldata/train_y.csv", "./mldata/train")
# print(df_train, df_val)

# df = load_to_dataframe("./mldata/train_y.csv", "./mldata/train", split = 0)
# print(df)


# df['image_filename'] = '/Users/malluin/goinfre/train/' + df['image_filename']
# df['image_filename'] = df['image_filename'].apply(load_img)
