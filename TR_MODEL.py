from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data_for_train import PREPARE_DATA_FOR_TRAIN
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from FEATURESELECTION import FeatureSelection
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import f1_score

class TrainModels:

    def normalize_data(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled


    def F1_score(self, y_test, predictions_proba, threshold):
        predictions_proba = pd.DataFrame(predictions_proba)
        predictions_proba.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
        threshold = threshold / 100
        try:
            predictions = []
            y_test_filtered = []
            for i in range(len(y_test)):
                if predictions_proba[1][i] > threshold:
                    predictions.append(1)
                    y_test_filtered.append(y_test['close'][i])
                elif predictions_proba[0][i] > threshold:
                    predictions.append(0)
                    y_test_filtered.append(y_test['close'][i])

            if not predictions:
                return 0, 0, 0
            
            # استفاده از f1_score از sklearn.metrics
            f1 = f1_score(y_test_filtered, predictions)
            # Calculate wins and losses
            
            wins = sum(1 for i in range(len(y_test_filtered)) if y_test_filtered[i] == predictions[i])
            loses = len(y_test_filtered) - wins

            return f1, wins, loses

        except Exception as e:
            print(f"Error in acc_by_threshold: {e}")
            return 0, 0, 0



    # def acc_by_threshold(self, y_test, predictions_proba, threshold):
    #     predictions_proba = pd.DataFrame(predictions_proba)
    #     predictions_proba.reset_index(inplace=True, drop=True)
    #     y_test.reset_index(inplace=True, drop=True)
    #     threshold = threshold / 100
    #     try:
    #         wins = 0
    #         loses = 0
    #         for i in range(len(y_test)):
    #             if predictions_proba[1][i] > threshold:
    #                 if y_test['close'][i] == 1:
    #                     wins += 1
    #                 else:
    #                     loses += 1
    #             if predictions_proba[0][i] > threshold:
    #                 if y_test['close'][i] == 0:
    #                     wins += 1
    #                 else:
    #                     loses += 1
    #         return (wins * 100) / (wins + loses), wins, loses
    #     except Exception as e:
    #         print(f"Error in acc_by_threshold: {e}")
    #         return 0, 0, 0

    def Train(self, data, depth, page, feature, qty, iterations, threshold, primitive_hours=[]):
        try:
            print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{qty}, iter:{iterations}, threshold:{threshold}, primitive_hours:{primitive_hours}")
            data = data[-qty:]
            data = TimeConvert().exec(data)
            data.reset_index(inplace=True, drop=True)
            primitive_hours = SelectTimeToDeleteOptimized().exec(data, primitive_hours)
            data, target, primitive_hours = PREPARE_DATA_FOR_TRAIN().ready(data, primitive_hours)
            data, target = PageCreatorParallel().create_dataset(data, target, page)
            primitive_hours = primitive_hours[page:]
            data, target = DeleteRow().exec(data, target, primitive_hours)
            fs = FeatureSelection()
            selected_data, selected_features = fs.select(data, target, feature)
            data = selected_data.copy()
            data = self.normalize_data(data)
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)
        except Exception as e:
            print(f"Error in preparing data: {e}")
            return 0,

        if iterations < 1000:
            iterations = 1000

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.005,
            depth=depth,
            loss_function='Logloss',
            verbose=100,
            task_type='CPU',
            random_state= 42,
            eval_metric='F1',
            early_stopping_rounds=100,
            l2_leaf_reg=5,
            subsample=0.9,
            bagging_temperature=1,
        )


        try:
            print(f"depth = {depth}, pages = {page}, iteration = {iterations}")
            print("Start training model")
            model.fit(X_train, y_train, eval_set=(X_test, y_test))  # اضافه کردن eval_set برای early stopping
            print("End of training model")
            predictions_proba = model.predict_proba(X_test)
            acc, wins, loses = self.F1_score(y_test, predictions_proba, threshold)
            print(f"ACC:{acc}, wins:{wins}, loses:{loses}")
        except Exception as e:
            print(f"Error in training model: {e}")
            del data, selected_data, X_train, X_test, y_train, target, selected_features
            gc.collect()
            return 0,

        try:
            if wins + loses >= 0.1 * ((qty * 10) / 100):
                # if acc / 100 > 0.6:
                del data, selected_data, X_train, X_test, y_train, target, selected_features
                gc.collect()
                return (acc,)
                # else:
                #     del data, selected_data, X_train, X_test, y_train, target, selected_features
                #     gc.collect()
                #     return 0,
            else:
                del data, selected_data, X_train, X_test, y_train, target, selected_features
                gc.collect()
                return 0,
        except Exception as e:
            print(f"Error processing task: {e}")
            del data, selected_data, X_train, X_test, y_train, target, selected_features
            gc.collect()
            return 0,

