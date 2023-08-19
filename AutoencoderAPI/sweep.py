from tqdm.notebook import tqdm
from random import choice, choices
from sklearn.model_selection import ParameterGrid
from .recurentTriplet import recurentTriplet



class sweep:

    def __init__(self) -> None:
        pass


    def grid_search(self, name, run_config, sweep_config):
        """
        # Sweep 

        sweep(name, test_number, config)

        Train a series of neural networks 

        Parameters
        ----------
        - name : str
                - Name of the folder created to store the runs.
        - run_config : dict
                - Dictionary containing the run parameters.
        - sweep_config : dict
                - Dictionary containing the sweep parameters.

        Returns
        -------
        - None
        """
        sweep_index = 0
        parameter_dict = {}
        run_config['sweep'] = True
        path_save = run_config['files']['path_save']
        for parameter in sweep_config['search_param']:
            parameter_dict[parameter[1]] = sweep_config[parameter[1]]
        parameter_list = list(ParameterGrid(parameter_dict))
        test_number = len(parameter_list)

        for parameters in tqdm(parameter_list, total=test_number, desc="Run"):
            for value in parameters:
                run_config['train'][value] = parameters[value]
            
            run_config['files']['path_save'] = f"{path_save}/{name}/run {str(sweep_index).rjust(len(str(test_number)), '0')}"
            exp = recurentTriplet()
            exp.run(run_config)
            sweep_index += 1



    def random_search(self,):
        """
        TODO
        update to current structure
        """

        if config['sweep']['search_type'] == 'random_search':
                pbar = tqdm(total=test_number)
                while config_run['internal']["sweep_index"] <= test_number:

                    if 'layer_size_possibility' in config['sweep']['search_param']:
                            layer_number = choice(config['run']['layer_number'])
                            layer_list = choices(config['sweep']['layer_size_possibility'], k = int(layer_number / 2))
                            layer_list.append(config['network']['output_dimension'])
                            
                            config_run['run']['layer_list'] = layer_list + list(reversed(layer_list))[1:]

                    for parameter in config['sweep']['search_param']:

                        if parameter == 'activation_possibilty':
                            activation_list = choices(config['sweep'][parameter], k = int(layer_number / 2))
                            config_run['run']['activation_list'] = activation_list + list(reversed(activation_list))[1:]
                        elif parameter == 'layer_size_possibility':
                            pass
                        else:
                            config_run['sweep'][parameter] = choice(config[parameter])
                        
                    # Train the newly created config
                    self.run(run_config)
                    config_run['internal']['sweep_index'] += 1
                    pbar.update(1)
                pbar.close()





        #              if config['sweep']['sweep_name'] is not None:
        #    log_path = f"{config['files']['path_save']}/{config['sweep']['sweep_name']}/sweep {str(config['internal']['sweep_index']).rjust(config['internal']['number_size'], '0')}"
        #else: