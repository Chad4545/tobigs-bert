import ujson
import os

FILE_DIR = 'data/train/high/'
file_list = os.listdir(FILE_DIR)
print(file_list)

for file_name in file_list:
    with open('{0}/{1}'.format(FILE_DIR, file_name)) as load_file:
        data = ujson.load(load_file)
        print(data)
        print(data['article'])
        print(data['options'])
        break