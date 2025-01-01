import torch.optim as optim
def build_optimizer(hparams):
    def optimizer(params):
        optim_method = hparams['optim_method']
        if optim_method == 'adam':
            return optim.Adam(
                params,
                lr = hparams['learning_rate'],
                weight_decay = hparams['weight_decay']
            )
        elif optim_method == 'sgd':
            return optim.SGD(
                params,
                lr=hparams['learning_rate'],
                weight_decay=hparams['weight_decay'],
                momentum=hparams['momentum']
            )
        else:
            raise NotImplementedError

    return optimizer