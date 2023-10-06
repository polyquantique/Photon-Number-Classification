from os import listdir
import numpy as np


def file_name_table(path, size, string_index):

    decibel_dict = {}
    for file_name in listdir(path):
        
        data = np.fromfile(f"{path}/{file_name}", dtype=np.float16).reshape(-1,size)
        decibel = file_name[string_index[0]:string_index[1]]
        
        if decibel in decibel_dict.keys():
            decibel_dict[decibel] = np.concatenate([decibel_dict[decibel], data])
        else:
            decibel_dict[decibel] = data

    return decibel_dict

        