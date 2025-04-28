import numpy as np
from sklearn.metrics import roc_auc_score
from utilities import obtain_tuned_model, downsampling, get_num_periods
from sklearn.preprocessing import RobustScaler
from typing import Literal
from utilities import MODEL_TYPES, DATASET_TYPES
from scipy.spatial.distance import directed_hausdorff
from utils_laf import EM, Dataset


class SelectionModel(object):
    def __init__(self, window_size: int, model_name: MODEL_TYPES, dataset: DATASET_TYPES):
        self.feature_list = []
        self.label_list = []
        self.model_pool = []
        self.scaler_pool: list[RobustScaler] = []

        self.dataset = dataset
        self.model_name = model_name
        self.window_size = window_size

        # temp storage
        self.last_method = ''
        self.model_ranking = []
        self.validation_period = 0
        self.validation_perf = []
        

    def train_new_model(self, X, y, model_name, period, type):
        model = obtain_tuned_model(model_name, self.dataset, period, type)
        model.fit(X, y)
        return model


    def initial_fit(self, X_list, y_list):
        self.feature_list = X_list.copy()
        self.label_list = y_list.copy()

        X = np.vstack(self.feature_list)
        y = np.hstack(self.label_list)
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        X, y = downsampling(X, y)

        self.model_pool.append(self.train_new_model(X, y, self.model_name, len(self.feature_list)+1, 'w'))
        self.scaler_pool.append(scaler)


    def fit(self, X, y):
        self.feature_list.append(X)
        self.label_list.append(y)

        X = np.vstack(self.feature_list[-self.window_size:])
        y = np.hstack(self.label_list[-self.window_size:])
        scaler = RobustScaler()
        X = scaler.fit_transform(X)
        X, y = downsampling(X, y)

        self.model_pool.append(self.train_new_model(X, y, self.model_name, len(self.feature_list)+1, 'w'))
        self.scaler_pool.append(scaler)

    def update_states(self, method, model_ranking, validation_period=None, validation_perf=None):
        self.last_method = method
        self.model_ranking = model_ranking
        self.validation_period = validation_period
        self.validation_perf = validation_perf

    def aggregate_proba(self, X, num_models=1, aggregation=None):
        if num_models == 1 or len(self.model_ranking) == 1:
            model_idx = self.model_ranking[0]
            return self.model_pool[model_idx].predict_proba(self.scaler_pool[model_idx].transform(X))[:, 1]
        
        model_indexes = self.model_ranking[:num_models]
        proba_predicted = np.array([self.model_pool[idx].predict_proba(self.scaler_pool[idx].transform(X))[:, 1] 
                                    for idx in model_indexes])
        if aggregation == 'average':
                return np.average(proba_predicted, axis=0)
        elif aggregation == 'voting':
            return np.average(proba_predicted > 0.5, axis=0)

    def predict_proba(self, X, y_oracle, method: 
                      Literal['stationary', 'retrain', 'oracle',
                              'laf', 'crc',
                              'temporal', 'temporal_rev',
                              'dist', 'dist_leak'
                              ]):

        # baseline: stationary (always use the initial model)
        if method == 'stationary':
            self.update_states(method, [0], -1, [-1])
            return self.aggregate_proba(X)
    
        # baseline: retrain (always use the newest model)
        if method == 'retrain':
            self.update_states(method, [len(self.model_pool)-1], -1, [-1])
            return self.aggregate_proba(X)
            
        if method == 'oracle':
            val_auc = [
                roc_auc_score(y_oracle, self.model_pool[idx].predict_proba(self.scaler_pool[idx].transform(X))[:, 1])
                for idx in range(len(self.model_pool))]
            model_ranking = np.argsort(val_auc)[::-1]
            self.update_states(method, model_ranking.tolist(), len(self.model_pool), val_auc)
            return self.aggregate_proba(X)

        # when only one model available, fallback to retraining
        if len(self.model_pool) == 1:
            self.update_states(method, [0], -1, [-1])
            return self.aggregate_proba(X)
        
        if method == 'crc':
            proba_predicted = np.array([self.model_pool[idx].predict_proba(self.scaler_pool[idx].transform(X))[:, 1] 
                                        for idx in range(len(self.model_pool))])
            confidence = np.maximum(proba_predicted, 1.0-proba_predicted)
            mean_confidence = np.sum(confidence, axis=1) / confidence.shape[1]
            model_ranking = np.argsort(mean_confidence)[::-1]
            self.update_states(method, model_ranking.tolist(), len(self.model_pool), [-1])
            return self.aggregate_proba(X)

        # LaF as a standalone mechanism
        if method == 'laf':
            # predictions of all models on the testing samples
            preds = np.array([self.model_pool[idx].predict(self.scaler_pool[idx].transform(X))
                              for idx in range(len(self.model_pool))])
            
            # filter out the samples with no discrimination
            filtering_idx = np.argwhere(np.logical_or(np.all(preds[..., :] == False, axis=0), np.all(preds[..., :] == True, axis=0)))
            filtered_preds = np.delete(preds, filtering_idx, axis=1)
            
            # use majority voting to calculate pseudo label
            pseudo_labels = np.sum(filtered_preds, axis=0) * 2 > filtered_preds.shape[0]

            # alpha and beta parameters
            sample_difficulty = np.sum(filtered_preds != pseudo_labels, axis=0) / filtered_preds.shape[0]
            model_estimated_acc = np.sum(filtered_preds == pseudo_labels, axis=1) / filtered_preds.shape[1]
            
            # call LaF EM algorithm to optimize beta
            data = Dataset()
            data.numClasses = 2
            data.priorZ = np.ones(data.numClasses) / data.numClasses
            data.priorZ[-1] = 1 - np.sum(data.priorZ[:-1])

            data.labels = filtered_preds
            data.priorAlpha = sample_difficulty
            data.priorBeta = model_estimated_acc

            data.numLabelers = filtered_preds.shape[1]
            data.numTasks = filtered_preds.shape[0]
            data.numLabels = data.numLabelers * data.numTasks

            data.probZ = np.empty((data.numTasks, data.numClasses))
            data.beta = np.empty(data.numTasks)
            data.alpha = np.empty(data.numLabelers)

            EM(data)

            # highest expectancy are better models
            model_ranking = np.argsort(data.beta)[::-1]
            self.update_states(method, model_ranking.tolist(), len(self.feature_list), [-1])
            return self.aggregate_proba(X)

        # temporal adjacency estimation:
        if method.startswith('temporal'):
            X_val = self.feature_list[-1]
            y_val = self.label_list[-1]
            
            val_auc = [roc_auc_score(y_val, self.model_pool[idx].predict_proba(self.scaler_pool[idx].transform(X_val))[:, 1])
                       for idx in range(len(self.model_pool))]
            model_ranking = np.argsort(val_auc[:-1])[::-1]

            if 'rev' in method and model_ranking[0] == len(self.model_pool) - 2:
                model_ranking = np.concatenate(([len(self.model_pool)-1], model_ranking))
            else:
                model_ranking = np.concatenate((model_ranking, [len(self.model_pool)-1]))

            self.update_states(method, model_ranking.tolist(), len(self.model_pool)-1, val_auc)
            return self.aggregate_proba(X)

        if method.startswith('dist'):
            if not self.last_method.startswith('dist'):
                # only calculate the distance between X and latter half of prev periods
                # as the model would data leakage on validation data before that
                feature_dists = [directed_hausdorff(self.feature_list[i], X)[0]
                                    for i in range(get_num_periods(self.dataset)//2, len(self.feature_list))]
                validation_period = np.argsort(feature_dists)[0] + get_num_periods(self.dataset)//2
            else:
                validation_period = self.validation_period

            X_val = self.feature_list[validation_period]
            y_val = self.label_list[validation_period]
            val_auc = [roc_auc_score(y_val, self.model_pool[idx].predict_proba(self.scaler_pool[idx].transform(X_val))[:, 1])
                       for idx in range(len(self.model_pool))]

            if 'leak' in method:
                model_ranking = np.argsort(val_auc)[::-1]
            else:
                safe_part = val_auc[:validation_period-get_num_periods(self.dataset)//2+1]
                leak_part = val_auc[validation_period-get_num_periods(self.dataset)//2+1:]
                model_ranking = np.concatenate((
                    np.argsort(safe_part)[::-1],
                    np.argsort(leak_part)[::-1] + len(safe_part)
                ))
            
            self.update_states(method, model_ranking.tolist(), validation_period, val_auc)
            return self.aggregate_proba(X)
