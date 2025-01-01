import torch
from .pre_train_model import PreTrainModel
from models.loss import CrossEntropyLabelSmooth
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def pre_train_model(backbone, configs, hparams, src_dataloader, avg_meter, logger, device):
    # model.
    pre_trained_model = PreTrainModel(backbone, configs, hparams)
    pre_trained_model = pre_trained_model.to(device)
    # optimizer
    pre_optimizer = torch.optim.Adam(
        pre_trained_model.network.parameters(),
        lr=hparams['pre_learning_rate'],
        weight_decay=hparams['weight_decay']
    )
    # loss.
    cross_entropy = CrossEntropyLabelSmooth(configs.num_classes, device, epsilon=0.1)

    # pretrain the model.
    for epoch in range(1, hparams['num_epochs'] + 1):
        pred_list = []
        label_list = []
        for step, (src_x, src_y, _) in enumerate(src_dataloader):
            # input src data
            if isinstance(src_x, list):
                src_x, src_y = src_x[0].float().to(device), src_y.long().to(device)  # list: (raw_data, aug1, aug2)
            else:
                src_x, src_y = src_x.float().to(device), src_y.long().to(device)  # raw_data

            # extract features
            src_pred = pre_trained_model(src_x)
            # calculate loss
            src_cls_loss = cross_entropy(src_pred, src_y)
            # optimizer zero grad
            pre_optimizer.zero_grad()
            # calculate gradient
            src_cls_loss.backward()
            # update weights
            pre_optimizer.step()

            # acculate loss
            avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # append prediction
            pred_list.extend(src_pred.argmax(dim=1).detach().cpu().numpy())
            label_list.extend(src_y.detach().cpu().numpy())

        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list, average='macro')
        print('source acc:', acc, 'source f1:', f1)

        # logging
        logger.debug(f'[Epoch : {epoch}/{hparams["num_epochs"]}]')
        for key, val in avg_meter.items():
            logger.debug(f'{key}\t: {val.avg:2.4f}')
        logger.debug(f'-------------------------------------')

    src_only_model = deepcopy(pre_trained_model.network.state_dict())

    return src_only_model, pre_trained_model


