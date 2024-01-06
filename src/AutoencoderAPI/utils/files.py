#General import
import pickle

#function specific import

def open_object(file_name):
    """
    Open a file using the pickle library. 

    Parameters
    ----------
    file_name : str
        Name of the file to open and read.

    Returns
    -------
    None
    
    """
    try:
        with open(file_name, 'rb') as f:
            object_ = pickle.load(f)
    except Exception as ex:
        print("Error when loading file : ", ex)

    return object_



#function specific import

def save_object(save_object, file_name):
    """
    Save an object to a .bin file using the pickle library.
    It is used in this context for dictionaries.

    Parameters
    ----------
    save_object : dict
        Dictionary to save.
    file_name : str
        Name of the file to save.

    Returns
    -------
    None
    """
    try:
        with open(f"{file_name}.bin", 'wb') as f:
            pickle.dump(save_object, f)
    except Exception as ex:
        print("Error during saving process : ", ex)



#function specific import
import torch

def save_all(log_path, network, results, loss, config):
    """
    Save all the elements available in `run` folders.
    The network is saved as a Pytorch object and other
    dictionaries are 

    Parameters
    ----------
    save_object : dict
        Dictionary to save.
    file_name : str
        sName of the file to save.

    Returns
    -------
    None
    """
    torch.save(network.state_dict() , f"{log_path}/model.pt")
    save_object(results , f"{log_path}/results")
    save_object(loss , f"{log_path}/loss")
    save_object(config, f"{log_path}/log")

# Function specific import
import zipfile
import os

def open_zip(files, folder_path):

    with zipfile.ZipFile(folder_path) as z:
        for filename in files:
            if not os.path.isdir(filename):
                # read the file
                with z.open(filename) as f:
                    for line in f:
                        print(line)
