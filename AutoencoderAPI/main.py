




def sweep(self, name, build_autoencoder, config, test_number=1):
    """
    # Sweep 

    sweep(name, test_number, config)

    Train a series of neural networks (change the activation functions and layer size) 
    using a search process (random search or grid search). 
    The activation function are defined to keep the autoencoder structure.
    Any parameter can be modified in the sweep to experiment.

    TODO
    - Find criteria for dimensionality reduction and add it to the sweep

    name (str) : 
    test_number (int) : Number of model to create and train.
    config (dict) : Configuration parameters to create the autoencoders.

    Parameters
    ----------
    - name : str
            - Name of the folder created to store the runs.
    - test_number : int
            - Number of model to create and train.
    - build_autoencoder : class
            - Pytorch neural network class with a `__init__` definition and `forward` process.
    - config : dict
            - Dictionary containing the experiment parameters. 
                See the `autoencoder` class for more details on the config dictionary.

    Returns
    -------
    - None
    """
    config['internal'] = {}
    config['sweep']['sweep_name'] = name
    config_run = config.copy()
    config_run['internal']['sweep_index'] = 0
    layer_number = config_run['run']['layer_number']


    # Creat random test
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
            self.run(build_autoencoder, config_run)
            config_run['internal']['sweep_index'] += 1
            pbar.update(1)
        pbar.close()

    elif config['sweep']['search_type'] == 'grid_search':
        parameter_dict = {}
        for parameter in config['sweep']['search_param']:
            parameter_dict[parameter[1]] = config[parameter[0]][parameter[1]]

        parameter_list = list(ParameterGrid(parameter_dict))
        test_number = len(parameter_list)
        config_run['internal']['number_size'] = len(str(test_number))

        pbar = tqdm(total=test_number)
        for parameters in parameter_list:
            for value in parameters:
                print(f"{value} : " , parameters[value])
                config_run['train'][value] = parameters[value]
            
            # Train the newly created config
            self.run(build_autoencoder, config_run)
            config_run['internal']['sweep_index'] += 1
            pbar.update(1)
        pbar.close()