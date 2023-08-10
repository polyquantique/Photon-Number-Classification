#General import
import pickle

#Class specific import

def open_object(file_name):
    """
    # open_object

    open_object(file_name)

    Open a file using the Pickle library. 

    Parameters
    ----------
    - file_name : str
        - Name of the file to open and read.

    Returns
    -------
    - 
    """
    try:
        with open(file_name, 'rb') as f:
            object_ = pickle.load(f)
    except Exception as ex:
        print("Error when loading file : ", ex)

    return object_


#Class specific import

def save_object(save_object, file_name):
        """
        # save_object

        save_object(save_object, file_name)

        Save an object to a .bin file using the Pickle library.
        Is used in this context for dictionaries.

        Parameters
        ----------
        - save_object : dict
                - Dictionary to save.
        - file_name : str
                - Name of the file to save.

        Returns
        -------
        None
        """
        try:
            with open(f"{file_name}.bin", 'wb') as f:
                pickle.dump(save_object, f)
        except Exception as ex:
            warnings.warn("Error during saving process : ", ex)


#Class specific import
import torch

def save_all(log_path, network, results, loss, config):
    torch.save(network.state_dict() , f"{log_path}/model.pt")
    save_object(results , f"{log_path}/results")
    save_object(loss , f"{log_path}/loss")
    save_object(config, f"{log_path}/log")




