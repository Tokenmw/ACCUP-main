import math

def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9
        }
        self.alg_hparams = {
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 0.0003, 'filter_K': 100, 'k_neighor': 3, 'tau': 1, 'lamb': 1.0, 'temperature': 0.6},
            'NoAdap': {'pre_learning_rate': 0.001}
        }

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9
        }

        self.alg_hparams = {
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 1e-5, 'filter_K': 50, 'k_neighor': 5, 'tau':50, 'lamb':1.0, 'temperature':0.3},
            'NoAdap' : {'pre_learning_rate': 0.001}
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method':'adam',
            'momentum':0.9
        }
        self.alg_hparams = {
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 0.0003, 'filter_K': 10, 'k_neighor': 5, 'tau':20, 'lamb': 0.1, 'temperature':0.7},
            'NoAdap': {'pre_learning_rate': 0.001}
        }
