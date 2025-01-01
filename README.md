# [KDD 2025] Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation
#### *by: Peiliang Gong, Mohamed Ragab, Min Wu, Zhenghua Chen, Yongyi Su, Xiaoli Li, Daoqiang Zhang* <br/> 

## Accepted in the [31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track](https://kdd2025.kdd.org).


## Requirmenets:
- Python3
- Pytorch==1.9
- Numpy==1.23.5
- scikit-learn==1.0
- Pandas==1.3.4
- skorch==0.10.0 
- openpyxl==3.0.7
- Wandb=0.12.7

## Datasets

### Available Datasets
We used three public datasets in this study. We also provide the **preprocessed** versions as follows:
- [SSC](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/UD1IM9)
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [MFD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)


## Training procedure

The experiments are organised in a hierarchical way such that:
- Several experiments are collected under one directory assigned by `--exp_name`.

### Training a model

To train a model:

```
python trainers/tta_trainer.py  --exp_name All_trg  \
                --da_method ACCUP \
                --dataset HAR \
                --backbone CNN \
                --num_runs 3 \
```

## Citation
If you found this work useful for you, please consider citing it.

```
@inproceedings{accup,
  author = {Gong, Peiliang and Ragab, Mohamed and Wu, Min and Chen, Zhenghua and Su, Yongyi and Li, Xiaoli},
  title = {Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation},
  booktitle={31st SIGKDD Conference on Knowledge Discovery and Data Mining - Research Track},
  year = {2025}
}
```


