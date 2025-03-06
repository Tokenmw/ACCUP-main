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
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 0.0003, 'filter_K': 100, 'tau': 1, 'temperature': 0.6},
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
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 1e-5, 'filter_K': 50, 'tau':50, 'temperature':0.3},
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
            'ACCUP': {'pre_learning_rate': 0.001, 'learning_rate': 0.0003, 'filter_K': 10, 'tau':20, 'temperature':0.7},
            'NoAdap': {'pre_learning_rate': 0.001}
        }
