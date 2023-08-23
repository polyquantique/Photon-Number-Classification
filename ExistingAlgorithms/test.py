from .PIKA import PIKA

config = {
    'Dataset_path'          : "Datasets/NIST PIKA 1 (4096)",
    'Dataset_signal_size'   : 4096,
    'Average_photon_number' : 1.49,
    'Epochs'                : 100,
    'Optimization_iter'     : 20
}

def PoissonUpdate():

    PIKA.run_PIKA(config)

    mu = 4.4
    nAdd = 3
    nDel = 2
