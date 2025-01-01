import sys

sys.path.append('../../ADATIME/')
import torch
import torch.nn.functional as F
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions

from torchmetrics import Accuracy, AUROC, F1Score
from dataloader.dataloader import data_generator, whole_targe_data_generator
from dataloader.demo_dataloader import data_generator_demo, whole_targe_data_generator_demo
from configs.data_model_configs import get_dataset_class
from configs.tta_hparams_new import get_hparams_class
from algorithms.get_tta_class import get_algorithm_class

from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from pre_train_model.build import pre_train_model
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

class TTAAbstractTrainer(object):
    """
   This class contain the main training functions for our method.
    """
    def __init__(self, args):
        self.da_method = args.da_method
        self.dataset = args.dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)

        self.run_description = f"{args.da_method}_{args.exp_name}"
        self.experiment_description = args.dataset

        self.home_path = os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.num_runs = args.num_runs
        self.dataset_configs, self.hparams_class = self.get_configs()
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method], **self.hparams_class.train_params}

        self.num_classes = self.dataset_configs.num_classes
        self.ACC = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.F1 = F1Score(task="multiclass", num_classes=self.num_classes, average="macro")
        self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)

    def sweep(self):
        pass

    def initialize_pretrained_model(self):
        backbone_fe = get_backbone_class(self.backbone)
        pretrained_model = PreTrainModel(backbone_fe, self.dataset_configs, self.hparams)
        pretrained_model = pretrained_model.to(self.device)

        return pretrained_model

    def pre_train(self):
        backbone_fe = get_backbone_class(self.backbone)
        # pretraining step
        self.logger.debug(f'Pretraining stage..........')
        self.logger.debug("=" * 45)
        non_adapted_model_state, pre_trained_model = pre_train_model(backbone_fe, self.dataset_configs, self.hparams, self.src_train_dl, self.pre_loss_avg_meters, self.logger, self.device)

        return non_adapted_model_state, pre_trained_model

    def evaluate(self, test_loader, tta_model):
        total_loss, preds_list, labels_list = [], [], []

        for data, labels, trg_idx in test_loader:
            if isinstance(data, list):
                data = [data[i].float().to(self.device) for i in range(len(data))]
            else:
                data = data.float().to(self.device)
            labels = labels.view((-1)).long().to(self.device)

            predictions = tta_model(data)
            loss = F.cross_entropy(predictions, labels)
            total_loss.append(loss.item())
            pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
            preds_list.append(pred)
            labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def get_tta_model_class(self):
        tta_model_class = get_algorithm_class(self.da_method)

        return tta_model_class

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")

        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")

        self.trg_whole_dl = whole_targe_data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams)

    def load_data_demo(self, src_id, trg_id, run_id = 0):
        self.src_train_dl = data_generator_demo(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator_demo(self.data_path, src_id, self.dataset_configs, self.hparams, "test")
        self.trg_whole_dl = whole_targe_data_generator_demo(self.data_path, trg_id, self.dataset_configs, self.hparams, seed_id = run_id)

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def calculate_metrics_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        # calculate metrics
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()

        risks = src_risk, trg_risk
        metrics = acc, f1, auroc

        return risks, metrics

    def save_tables_to_file(self, table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir, f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, non_adapted):
        save_dict = {
            "non_adapted": non_adapted
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, f"checkpoint.pt")
        torch.save(save_dict, save_path)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        pretrained_model = checkpoint['non_adapted']

        return pretrained_model

    def calculate_avg_std_wandb_table(self, results):
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}

        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        return results, summary_metrics

    def log_summary_metrics_wandb(self, results, risks):

        # Calculate average and standard deviation for metrics
        avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]

        avg_risks = [np.mean(risks.get_column(risk)) for risk in risks.columns[2:]]
        std_risks = [np.std(risks.get_column(risk)) for risk in risks.columns[2:]]

        # Estimate summary metrics
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        summary_risks = {risk: np.mean(risks.get_column(risk)) for risk in risks.columns[2:]}

        # append avg and std values to metrics
        results.add_data('mean', '-', *avg_metrics)
        results.add_data('std', '-', *std_metrics)

        # append avg and std values to risks
        results.add_data('mean', '-', *avg_risks)
        risks.add_data('std', '-', *std_risks)

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)

    def calculate_metrics(self, tta_model):

        self.evaluate(self.trg_whole_dl, tta_model)
        acc = self.ACC(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        f1 = self.F1(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        auroc = self.AUROC(self.full_preds.cpu(), self.full_labels.cpu()).item()
        trg_risk = self.loss.item()

        return acc, f1, auroc, trg_risk

    def calculate_risks(self):
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()

        return src_risk, trg_risk

    def append_results_to_tables(self, table, scenario, run_id, metrics):

        # Create metrics and risks rows
        if isinstance(metrics, float):
            results_row = [scenario, run_id, metrics]
        elif isinstance(metrics, tuple):
            results_row = [scenario, run_id, *metrics]

        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)

        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)

        return table

    def add_mean_std_table(self, table, columns):
        # Calculate average and standard deviation for metrics
        avg_metrics = [table[metric].mean() for metric in columns[2:]]
        std_metrics = [table[metric].std() for metric in columns[2:]]

        # Create dataframes for mean and std values
        mean_metrics_df = pd.DataFrame([['mean', '-', *avg_metrics]], columns=columns)
        std_metrics_df = pd.DataFrame([['std', '-', *std_metrics]], columns=columns)

        # Concatenate original dataframes with mean and std dataframes
        table = pd.concat([table, mean_metrics_df, std_metrics_df], ignore_index=True)

        # Create a formatting function to format each element in the tables
        format_func = lambda x: f"{x:.4f}" if isinstance(x, float) else x

        # Apply the formatting function to each element in the tables
        table = table.applymap(format_func)

        return table