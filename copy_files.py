import pandas as pd
from shutil import copyfile

df_copy = pd.read_csv('/home/seldon/mauna_kea/mldata/train_y_add.csv')
source_folder = '/home/seldon/mauna_kea/mldata/test_set/'
target_folder = '/home/seldon/mauna_kea/mldata/train_set/'

for file in df_copy['image_filename']:
	copyfile(source_folder + file, target_folder + 'test_' + file)
	# break;

df_copy['image_filename'] = 'test_' + df_copy['image_filename']
df_copy.to_csv('/home/seldon/mauna_kea/mldata/train_y_addtest.csv', index = None, header=True)
