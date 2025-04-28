# AIOps Evolution - Supplemental Materials
This repository contains the replication package for our study "Can We Recycle Our Old Models? An Empirical Evaluation of Model Selection Mechanisms for AIOps Solutions".

## Introduction
We organize the replication package into the following directories:
1. Experiments: this folder contains code for our main experiment (i.e., evaluating model selection mechanisms in AIOps);
2. Results data: this folder contains the results from experiments;
3. Results analysis: this folder contains our code for analyzing the experiment results and making plots.

## Experiments
This part contains code for our main experiment in evaluating various model updating approaches. All code could be found under the `experiment` folder.

### Experiment prerequisites
1. Dependencies: we compiled the required dependencies for running the Python experiment code in the `requirements.txt` under the root directory.
2. Datasets: We provide the datasets on the release page, please unzip and place them into the same folder as the experiment code.
3. Hyperparameter settings: We provide the setting files inside the experiment folder. We tune the hyperparameter on each sliding window in advance to save time when repeating our experiment 100 times.

### Execution
The experiment code accepts the following command-line arguments to select model, dataset, and iteration rounds for maximum flexibility.
1. `-d` is a **required** parameter for choosing the dataset. Two choices are available: `g` for the Google dataset, `b` for the Backblaze dataset.
2. `-m` is a **required** parameter for choosing the model. Five choices are available: `lr`, `cart`, `rf`, `gbdt`, and `nn`. Please note that the argument should be all *lowercase* letters.
3. `-n` is an optional parameter for the repetition time of the experiments. The default value is 100 iterations, which is also the same iteration number we used in our paper.
4. `-s` is an optional parameter for the starting round of the experiment. It would be useful if you would like to resume experiment from a specific round.

As some experiments could take a prolonged time to finish, we recommend executing them on a server with tools like `GNU Screen` or `nohup`. An example of evaluating the ensemble approaches on the `Google` data set and `RF` model in `100` iteration with `nohup` in the `background` and dump the command line output to `log_selection_g_rf.out` would be: `nohup python -u evaluate_selection_models.py -d g -m rf -n 100 > log_selection_g_rf.out 2>&1 &`.

## Experiment Results Data
This part contains the output data from our main experiments. All output CSV files could be found under the `results` folder.
We organize the CSV files into two folders. The `preliminary_results` folder contains files for the Preliminary Study section, while the `experiment_results` folder contains files for our main results, we only provide the CSV files after combining separate files together for simplicity.

## Results Analysis
This part contains code for the analysis of our experiment results. All code could be found under the `analysis` folder.

We have the following code available:
- `analyze_ranking.py` calculate the statistics for the ranking performance. A readily available output of this code can be found in `results/ranking_analysis.csv`.
- `result_analysis.R` contains code for plotting results figures and tables in our main experiment results.
