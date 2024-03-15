from os import listdir
import numpy as np


def file_name_table_bin(path, size, string_index):

    decibel_dict = {}
    for file_name in listdir(path):
        
        data = np.fromfile(f"{path}/{file_name}", dtype=np.float16).reshape(-1,size)
        decibel = file_name[string_index[0]:string_index[1]]
        
        if decibel in decibel_dict.keys():
            decibel_dict[decibel] = np.concatenate([decibel_dict[decibel], data])
        else:
            decibel_dict[decibel] = data

    return decibel_dict


def file_name_table_npy(path_data, path_dB, size, mean, std):

    decibel_dict = {} 
    decibel_dict_zeros = {} 
    number_file = len(listdir(path_data))

    dB = np.load(path_dB)

    for file_number in range(number_file):
        
        data = np.load(f"{path_data}/TracesNr{file_number}.npy").reshape(-1,size)
        
        data = data[:, 3250:4500]
        data = (data - mean)/std
        zeros = data[np.min(data, axis=1) >= 1.6]
        data = data[np.min(data, axis=1) < -0.5]
        data = data[np.max(data, axis=1) > 1]
        
        decibel = dB[file_number]
        
        if decibel in decibel_dict.keys():
            decibel_dict[decibel] = np.concatenate([decibel_dict[decibel], data])
            decibel_dict_zeros[decibel] = np.concatenate([decibel_dict_zeros[decibel], zeros])
        else:
            decibel_dict[decibel] = data
            decibel_dict_zeros[decibel] = zeros

    return decibel_dict, decibel_dict_zeros



def decibel_table_npy(path_data, path_dB, size, mean, std):

    decibel = [] 
    decibel_zeros = []
    data = []
    number_file = len(listdir(path_data))

    dB = np.load(path_dB)

    for file_number in range(number_file):
        
        data_temp = -1*np.load(f"{path_data}/TracesNr{file_number}.npy").reshape(-1,size)#-1*
        
        data_temp = data_temp[:, 3250:4500]
        data_temp = (data_temp - mean)/std
        zeros = data_temp[np.max(data_temp, axis=1) <= 0]
        data_temp = data_temp[np.min(data_temp, axis=1) < -0.55]
        data_temp = data_temp[np.max(data_temp, axis=1) > 0]


        decibel_zeros.append(dB[file_number] * np.ones(len(zeros)))
        decibel.append(dB[file_number] * np.ones(len(data_temp)))
        data.append(data_temp)

    decibel_zeros = np.concatenate(decibel_zeros)
    decibel = np.concatenate(decibel)
    data = np.concatenate(data)
        
    return data, decibel, decibel_zeros




def decibel_table_bin(path, size, string_index, dB, weights):

    X = []
    decibel = []
    files = listdir(path)
    file_weight = [int(weights[dB.index(i[67:71])] * 1_024) for i in files]

    for  w, file_name in zip(file_weight, files):
        
        data_temp = np.fromfile(f"{path}/{file_name}", dtype=np.float16).reshape(-1,size)[:w]
        decibel_temp = file_name[string_index[0]:string_index[1]]
        
        X.append(data_temp)
        decibel.append(float(decibel_temp) * np.ones(len(data_temp)))

    X = np.concatenate(X)
    decibel = np.concatenate(decibel)

    return X, decibel
