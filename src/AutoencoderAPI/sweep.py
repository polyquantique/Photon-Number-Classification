from tqdm.notebook import tqdm
from random import choice, choices
from sklearn.model_selection import ParameterGrid
#from .recurrentTriplet import recurrentTriplet
#from .transformer import transformer
from .fileBatch import fileBatch



class sweep:

    def __init__(self) -> None:
        pass


    def grid_search(self, name, run_config, sweep_config):
        """
        # Sweep 

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
        path_save = run_config['files']['path_save']
        name = sweep_config['sweep_name']

        run_config['sweep'] = True

        for parameter in sweep_config['search_param']:
            parameter_dict[parameter[1]] = sweep_config[parameter[1]]
        parameter_list = list(ParameterGrid(parameter_dict))
        test_number = len(parameter_list)

        for parameters in tqdm(parameter_list, total=test_number, desc="Run"):
            for value in parameters:
                run_config['train'][value] = parameters[value]
            
            run_config['files']['path_save'] = f"{path_save}/{name}/run {str(sweep_index).rjust(len(str(test_number)), '0')}"
            exp = fileBatch()
            exp.run(run_config)
            sweep_index += 1


    
    def random_search(self,run_config, sweep_config, sweep_number):
        """
        TODO
        update to current structure
        
        """
        path_save = run_config['files']['path_save']
        name = sweep_config['sweep_name']
        layer_number = len(run_config['network']['layer_list'])

        run_config['sweep'] = True
        sweep_index = 0

        for index in range(sweep_number):

            if ('network','layer_size_possibility') in sweep_config['search_param']:
                layer_list = choices(run_config['sweep']['layer_size_possibility'], k = layer_number // 2)                
                run_config['run']['layer_list'] = layer_list + [1] + list(reversed(layer_list))[1:]

            for (category, parameter) in sweep_config['search_param']:

                if parameter == 'activation_possibilty':
                    activation_list = choices(sweep_config['activation_possibilty'], k = layer_number // 2)
                    run_config['network']['activation_list'] = activation_list + list(reversed(activation_list))[1:]
                elif parameter == 'layer_size_possibility':
                    pass
                else:
                    run_config[category][parameter] = choice(sweep_config[parameter])
                
            # Train the newly created config
            run_config['files']['path_save'] = f"{path_save}/{name}/run {str(sweep_index).rjust(len(str(index)), '0')}"
            exp = fileBatch()
            exp.run(run_config)
            sweep_index += 1




