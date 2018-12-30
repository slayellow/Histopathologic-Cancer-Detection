from PIL import Image
import numpy as np
import os
import pandas as pd

class DataLoad():


    def __init__(self):
        self.label_dic = {}
        self.image_data_list = []
        self.image_name_list = []

    def open_csv(self, filepath):
        data = pd.read_csv(filepath, sep=",", dtype='unicode')  # Data Load
        train_label = data["id"]  # Train Data의 Label 값 저장
        train_data = data["label"]  # Train Data의 값들 저장
        for i in range(len(train_label)):
            self.label_dic[train_label[i]] = train_data[i]

    def search_file(self, filepath):
        for (path, dir, files) in os.walk(filepath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                file_name = os.path.splitext(filename)[0]
                if ext =='.tif':
                    self.open_file(path+'/'+filename, file_name)

    def open_file(self, filepath, file_name):
        print("Open file : " + file_name)
        im = Image.open(filepath)
        imarray = np.array(im)
        self.image_data_list.append(imarray)
        self.image_name_list.append(file_name)
        print('Finish file load')
