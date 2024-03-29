# Drift Classifier for our PoPETs 2024.2 Paper "MixMatch"

Drift classifier for our PoPETs 2024.2 paper "MixMatch: Flow Matching for Mixnet Traffic", a new deep learning architecture to conduct flow matching against Nym traffic.

Primary author of this artifact: [Marc Juarez](https://mjuarezm.github.io/).

Secondary author of this artifact: [Lennart Oldenburg](https://lennartoldenburg.de/).

This repository is part of a larger list of repositories that we make available as artifacts of our paper. Please find more detailed documentation (including steps to set up the surrounding directories that this repository expects to be in place already) in our [main paper repository](https://github.com/mixnet-correlation/mixmatch-flow-matching-for-mixnet-traffic_popets-2024-2).

Running the analysis process of this classifier in full requires access to powerful hardware (including, at best, a powerful GPU with sufficient video memory) and takes at least one day per dataset.


## Setting Up

Run the following steps to get started:
```bash
root@ubuntu2204 $   apt-get install --yes tree tmux
root@ubuntu2204 $   mkdir -p ~/mixmatch
root@ubuntu2204 $   cd ~/mixmatch
root@ubuntu2204 $   git clone https://github.com/mixnet-correlation/mixmatch_drift_classifier.git
root@ubuntu2204 $   cd mixmatch_drift_classifier
root@ubuntu2204 $   ./setup_1_conda.sh
root@ubuntu2204 $   ~/miniconda3/bin/conda init bash   # Activate conda, modify if you use a different shell
root@ubuntu2204 $   exit
```

Log back into instance `ubuntu2204` and run:
```bash
root@ubuntu2204(base) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(base) $   ./setup_2_conda-packages.sh
... This will take at least 20min to complete ...
root@ubuntu2204(base) $   conda env list
# conda environments:
#
base                  *  /root/miniconda3
mixmatch                 /root/miniconda3/envs/mixmatch
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   conda env list
# conda environments:
#
base                     /root/miniconda3
mixmatch              *  /root/miniconda3/envs/mixmatch
```

For below steps, we expect the following file system structure to be in place and populated with the respective files in the correct places:
```bash
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch
root@ubuntu2204(mixmatch) $   tree
.
├── datasets
│   └── baseline
│       ├── ... MANY ENTRIES
├── delay_matrices
│   ├── baseline
│   │   ├── test_delay_matrix.npz
│   │   ├── train_delay_matrix.npz
│   │   └── val_delay_matrix.npz
│   ├── high-delay
│   │   ├── test_delay_matrix.npz
│   │   ├── train_delay_matrix.npz
│   │   └── val_delay_matrix.npz
│   ├── live-nym
│   │   ├── test_delay_matrix.npz
│   │   ├── train_delay_matrix.npz
│   │   └── val_delay_matrix.npz
│   ├── low-delay
│   │   ├── test_delay_matrix.npz
│   │   ├── train_delay_matrix.npz
│   │   └── val_delay_matrix.npz
│   └── no-cover
│       ├── test_delay_matrix.npz
│       ├── train_delay_matrix.npz
│       └── val_delay_matrix.npz
├── mixmatch_drift_classifier
│   ├── ... MANY ENTRIES
```


## Prepare Datasets by Parsing Them

Exemplarily for dataset `baseline`, we go through the steps parsing, training, evaluating, and score calculating that make up the process of analyzing a dataset with this classifier. Follow similar steps for the remaining datasets.
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   python parse.py ../datasets/baseline --delaymatpath ../delay_matrices/baseline --experiment 1
... Takes at least 20min to complete ...
```


## Train on Parsed Dataset

```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python train.py
... Takes on the order of many hours complete ...
```

When running the drift classifier on multiple datasets, we recommend to name data and results folders within `~/mixmatch/deeplearning/mixmatch_drift_classifier` explicitely after their respective experiment/dataset/purpose.

For any of these high-level steps, you can always check the produced logs:
```bash
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   ls -lahtr ./results/latest/
```


## Evaluate a Trained Model

```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/latest/ ./results/latest/
... Takes on the order of some hours to complete ...
```

For the special case of the `two-to-one` experiment that is based on the `baseline` dataset, we start from the `baseline`-trained model and instruct the model at inference time to build and analyze the `two-to-one` dataset ad-hoc in the following way:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python get_scores.py ./data/A_BASELINE_DATA_FOLDER/ ./results/A_BASELINE_RESULTS_FOLDER/ --two2one_case1   # Semi-matched case, use '--two2one_case2' instead for the unmatched setting
... Takes on the order of some hours to complete ...
```


## Calculate ROC Scores

```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/latest/
... Takes on the order of 1 hour to complete ...
```

For the special case of the `two-to-one` experiment, run:
```bash
root@ubuntu2204(base) $   tmux
root@ubuntu2204(base) $   conda activate mixmatch
root@ubuntu2204(mixmatch) $   cd ~/mixmatch/mixmatch_drift_classifier
root@ubuntu2204(mixmatch) $   TF_CPP_MIN_LOG_LEVEL=3 TF_DETERMINISTIC_OPS=1 PYTHONHASHSEED=0 python calculate_roc.py ./results/A_BASELINE_RESULTS_FOLDER/ --two2one
... Takes on the order of 1 hour to complete ...
```
