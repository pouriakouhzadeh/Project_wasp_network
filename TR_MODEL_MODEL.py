from timeconvert import TimeConvert
from selecttimetodelete import SelectTimeToDeleteOptimized
from preparing_data_for_train import PREPARE_DATA_FOR_TRAIN
from PAGECREATOR import PageCreatorParallel
from deleterow import DeleteRow
from feature_selection_for_TMM import FeatureSelection_for_TMM
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.metrics import f1_score

class TrainModelsReturn:

    def normalize_data(self, data):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data_scaled, scaler


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

    def Train(self, data, depth, page, feature, QTY, iterations, Thereshhold, primit_hours=[]):
        try :
            print(f"depth:{depth}, page:{page}, features:{feature}, QTY:{QTY}, iter:{iterations}, Thereshhold:{Thereshhold}, primit_hours:{primit_hours}")
            data = data[-QTY:]
            data = TimeConvert().exec(data)        
            data.reset_index(inplace=True, drop=True)
            primit_hours = SelectTimeToDeleteOptimized().exec(data, primit_hours)
            data, target, primit_hours = PREPARE_DATA_FOR_TRAIN().ready(data, primit_hours)
            data, target = PageCreatorParallel().create_dataset(data, target, page)
            primit_hours = primit_hours[page:]
            data, target = DeleteRow().exec(data, target, primit_hours)
            fs = FeatureSelection_for_TMM()
            selected_data, selected_features, features_indicts = fs.select(data, target, feature)
            data = selected_data.copy()
            data, scaler = self.normalize_data(data)
            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.1, random_state=1234)
        except :
            print("Error in preparing data .... program rised")
            return(0,0,0, )
        # print(f"primit_hours  = {primit_hours}")
        if iterations < 1000 :
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

        try :
            print("Start training model")
            model.fit(X_train, y_train)
            predictions_proba = model.predict_proba(X_test)
            acc, wins, loses = self.F1_score(y_test, predictions_proba, Thereshhold)
            if acc > 0.78 :
                print("END of Training model successfully , return model and parameters")
                return model, features_indicts, scaler, acc
            else :
                print(f"acc is {acc} and this is less than 0.78")
                return(0, 0, 0, )
        except :
            print("Error in training model ... program rised")
            return(0, 0, 0, )
