import os
import timeit
import argparse
import numpy as np
import pandas as pd
from utilities import obtain_period_data, obtain_metrics, get_dataset_name
from selection_model import SelectionModel


import warnings
warnings.filterwarnings('ignore')

OUTPUT_FILE_PREFIX = r'selection'
RANDOM_CONST = 114514


def experiment_driver(feature_list, label_list, dataset, model_type, out_file, n_round):
    out_columns =  [
        'Scenario', 'Model', 'Round', 'Testing Period', 
        'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B',
        'Training Time', 'Testing Time', 'Validation Period', 'Model Ranking', 'Validation Perf'
    ]
    out_ls = []

    selection_methods = [
        'stationary', 'retrain', 'oracle',
        'laf', 'crc',
        'temporal', 'temporal_rev',
        'dist', 'dist_leak'
    ]

    num_periods = len(feature_list)
    print('Total number of periods:', num_periods)
    print('Building models from first', num_periods//2, 'periods')

    # initial training features and labels (first sliding window)
    X_train = feature_list[: num_periods//2]
    y_train = label_list[: num_periods//2]

    model = SelectionModel(num_periods//2, model_type, dataset)
    
    np.random.seed(RANDOM_CONST+n_round*num_periods+num_periods//2-1)
    print('Doing initial training...')
    start_time = timeit.default_timer()
    model.initial_fit(X_train, y_train)
    training_time = timeit.default_timer() - start_time
    print('Done initial training.')

    for i in range(num_periods//2, num_periods):
        X_test = feature_list[i]
        y_test = label_list[i]

        print('Testing models on period', i + 1)
        for method in selection_methods:
            start_time = timeit.default_timer()
            testing_probas = model.predict_proba(X_test, y_test, method)
            testing_time = timeit.default_timer() - start_time
            out_ls.append(
                [method, model_type, n_round, i + 1] + 
                obtain_metrics(y_test, testing_probas) +
                [training_time, testing_time, model.validation_period, model.model_ranking, model.validation_perf]
            )
        
        out_df = pd.DataFrame(out_ls[-len(selection_methods):], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))

        if i + 1 == num_periods: # skip fitting for the last period
            continue

        print('Fitting models on period', i + 1)
        start_time = timeit.default_timer()
        np.random.seed(RANDOM_CONST+n_round*num_periods+i)
        model.fit(X_test, y_test)
        training_time += timeit.default_timer() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment on concept drift detection approaches')
    parser.add_argument("-d", help="specify the dataset, g=Google, b=Backblaze, and a=Alibaba.", required=True, choices=['g', 'b', 'a'])
    parser.add_argument("-m", help="specify the model type.", required=True, choices=['lr', 'cart', 'gbdt', 'nn', 'rf'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    parser.add_argument("-s", help="starting from this round.")
    args = parser.parse_args()

    n_rounds = int(args.n)
    dataset = args.d
    model = args.m
    feature_list, label_list = obtain_period_data(dataset)
    start_round = 0
    if args.s != None:
        start_round = int(args.s)

    output_path = f'{OUTPUT_FILE_PREFIX}_{get_dataset_name(dataset).lower()}_{model}.csv'
    print(f'Choose {get_dataset_name(dataset)} as dataset')

    # remove previously existing output file
    if os.path.isfile(output_path): 
        os.remove(output_path)
    print('Output path:', output_path)

    for i in range(start_round, n_rounds):
        print(f'Round {i+1}/{n_rounds}')
        experiment_driver(feature_list, label_list, dataset, model, output_path, i)
  
    print('Experiment completed!')
