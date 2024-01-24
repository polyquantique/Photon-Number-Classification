import torch
import numpy as np
from os import listdir

from .utils.files import open_object
from .setup.networks.autoencoder import build_autoencoder
from .fileBatchStack import fileBatchStack


class encoderStack:


    def __init__(self):
        """
        
        Parameters
        ----------

        models : list
            List of string containing the paths to the models used for the encoders.
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
    def train(self, config,
                stack_name,
                step_number,
                bw,
                flip = False):


        experts = [f'expert {step}' for step in range(step_number)]
        config['files']['stack'] = stack_name
        config['bw'] = bw
        config['flip'] = flip
        config['stack'] = True

        for index, step in enumerate(experts):

            try:
                if step in listdir(f"{config['files']['path_save']}/{stack_name}"):
                    continue
            except:
                pass

            config['files']['run_name'] = step
            config['models'] = experts[:index]

            func = fileBatchStack()
            func.run(config)
            

            

    
    def predict(self,):
        pass



    def stack(self, X, 
              path_experts):
        """
        Load all models given and labels input samples based on the encoder stack.

        Parameters
        ----------

        models : list
            List of string containing the paths to the models used for the encoders.
        """

        number_sample = len(X)
        output_label = torch.zeros(number_sample).int().to(self.device)
        index_samples = torch.range(0,number_sample-1).int().to(self.device)
        number_expert = len(listdir(path_experts))
        models = [f'expert {step}' for step in range(number_expert)]

        
        for index, model in enumerate(models):

            path = f"{path_experts}/{model}/fold 0"
            config_load = open_object(f"{path}/log.bin")
            space_separation = open_object(f"{path}/space.bin")
            size_network = config_load['internal']['size_network']

            network = build_autoencoder(config_load).float().to(self.device)
            network.load_state_dict(torch.load(f"{path}/model.pt"))

            network.eval()
            with torch.no_grad():
                X_pytorch = torch.from_numpy(X).view(-1, 1, size_network).float().to(self.device)
                X_low_dim = network(X_pytorch, encoding=True)
                #X_low_dim = X_low_dim#.numpy().reshape(-1, 1)
                #X_low = self.flip * X_low

            labels = torch.searchsorted(space_separation, X_low_dim).int().to(self.device)
            condition = labels == 0
            not_condition = torch.logical_not(condition)
            
            output_label[torch.masked_select(index_samples, condition)] = index
            
            index_samples = torch.masked_select(index_samples, not_condition)
            X = torch.masked_select(X, not_condition)
            
        return X, output_label




