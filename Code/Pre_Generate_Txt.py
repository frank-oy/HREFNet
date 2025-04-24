# -*- coding: utf-8 -*-
from os.path import exists, join
from os import listdir
import numpy as np
import random
import fileinput
import os
import re
"""
The purpose of this code is to generate ".txt" files in */TXT/ for training and testing
"""
def Get_file_list(image_path):

    files = os.listdir(image_path)

    files.sort(key=lambda x: int(re.sub(r'\D', '', x.split('.')[0])))
    return files, len(files)

def Generate_Txt(image_path, txt_name):
    f = open(txt_name, "w")
    files, files_num = Get_file_list(image_path)
    index_count = 0
    count = 0
    for file in files:
        index_count = index_count + 1
        if count == files_num - 1:
            f.write(image_path + str(file))
            break
        if index_count >= 0:
            f.write(image_path + str(file) + "\n")
            count = count + 1
    f.close()
    print("2 Finish Generate_Txt: ", txt_name)
