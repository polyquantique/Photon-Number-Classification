import pickle

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


def save_object(save_object, file_name) -> None:
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


    