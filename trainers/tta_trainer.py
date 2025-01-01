import sys

sys.path.append('../ADATIME')

import os
import pandas as pd
import torch
import collections
import argparse
import warnings
import sklearn.exceptions
from datetime import datetime
import numpy as np

from utils.utils import fix_randomness, starting_logs, AverageMeter
from tta_abstract_trainer import TTAAbstractTrainer
from optim.optimizer import build_optimizer

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

class TTATrainer(TTAAbstractTrainer):
    """
   This class contain the main training functions for our method.
    """
    def __init__(self, args):
        super(TTATrainer, self).__init__(args)
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        self.load_pretrained_checkpoint = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{'NoAdap'}_{'All_Trg'}")
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.summary_f1_scores = open(self.exp_log_dir + '/summary_f1_scores.txt', 'w')

    def test_time_adaptation(self):
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        for src_id, trg_id in self.dataset_configs.scenarios:
            cur_scenario_f1_ret = []
            for run_id in range(self.num_runs):
                self.run_id = run_id
                fix_randomness(run_id)
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, run_id)

                ## calculate class frequency of all target datasets. ##
                print('Total test datasize:', len(self.trg_whole_dl.dataset))
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for batch_idx, (inputs, target, _) in enumerate(self.trg_whole_dl):
                    for id in range(target.shape[0]):
                        all_labels[target[id]] += 1
                print('trg whole labels:', all_labels)
  

                ## pretraining from scratch.. ##
                non_adapted_model_state, pre_trained_model = self.pre_train()
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)

                ## if finshed pre_train. we can directly load pretrainModel from checkpoint ###
                # load_pretrained_checkpoint_path = os.path.join(self.load_pretrained_checkpoint, src_id + "_to_" + trg_id + "_run_" + str(run_id))
                # pre_trained_model = self.initialize_pretrained_model()
                # pre_trained_model_chk = self.load_checkpoint(load_pretrained_checkpoint_path)  # all method load same pretrained model.
                # pre_trained_model.network.load_state_dict(pre_trained_model_chk)

                optimizer = build_optimizer(self.hparams)
                if self.da_method == 'NoAdap':
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)
                tta_model.to(self.device)
                pre_trained_model.eval()

                metrics = self.calculate_metrics(tta_model)
                cur_scenario_f1_ret.append(metrics[1])
                scenario = f"{src_id}_to_{trg_id}"
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics[:3])
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, metrics[-1])

            cur_avg_f1_scores, cur_std_f1_scores = 100. * np.mean(cur_scenario_f1_ret), 100. * np.std(cur_scenario_f1_ret)
            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(scenario, ' : ', np.around(cur_avg_f1_scores, 2), '/', np.around(cur_std_f1_scores, 2), sep='', file=self.summary_f1_scores)

        # Calculate and append mean and std to tables
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        self.save_tables_to_file(table_results, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') +'_results')
        self.save_tables_to_file(table_risks, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') +'_risks')

        self.summary_f1_scores.close()

if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'E:\Dataset', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='HAR', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()
    trainer = TTATrainer(args)
    trainer.test_time_adaptation()
