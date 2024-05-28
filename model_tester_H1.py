import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LogisticRegression, Perceptron, Ridge, SGDClassifier,
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
import logging
import warnings

# Silent alerts --------------------------------------------
warnings.filterwarnings("ignore")

# Making objects -------------------------------------------

class ModelTester:
    def __init__(self, data, target, test_size=0.2, random_state=42):
        self.data = data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'BaggingClassifier': BaggingClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'VotingClassifier': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB())]),
            'SVC': SVC(),
            'LinearSVC': LinearSVC(),
            'NuSVC': NuSVC(),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'ExtraTreeClassifier': ExtraTreeClassifier(),
            'GaussianNB': GaussianNB(),
            'BernoulliNB': BernoulliNB(),
            'MLPClassifier': MLPClassifier(),
            'XGBClassifier': XGBClassifier(),
            'LGBMClassifier': LGBMClassifier(),
            'Perceptron': Perceptron(),
            'RidgeClassifier': RidgeClassifier(),
            'SGDClassifier': SGDClassifier(),
            'CalibratedClassifierCV': CalibratedClassifierCV(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
            'GaussianProcessClassifier': GaussianProcessClassifier(kernel=RBF()),
            'LinearSVC': LinearSVC(),
            'NuSVC': NuSVC(probability=True),
            'RidgeClassifierCV': RidgeClassifierCV(),
            'SGDClassifier': SGDClassifier(),
            'CalibratedClassifierCV': CalibratedClassifierCV(),
            'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
            'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
            'GaussianProcessClassifier': GaussianProcessClassifier(kernel=RBF()),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'MLPClassifier': MLPClassifier(),
            'SVC': SVC(probability=True),
            'KNeighborsClassifier': KNeighborsClassifier(),
            'RadiusNeighborsClassifier': RadiusNeighborsClassifier(),
            'VotingClassifier': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'RandomForestClassifier_2': RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=self.random_state),
            'GradientBoostingClassifier_2': GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, random_state=self.random_state),
            'AdaBoostClassifier_2': AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=self.random_state),
            'BaggingClassifier_2': BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=self.random_state),
            'ExtraTreesClassifier_2': ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=self.random_state),
            'VotingClassifier_2': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'SVC_2': SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=self.random_state),
            # اضافه کردن مدل‌های دیگر
            'SVC_3': SVC(kernel='poly', degree=3, C=1, gamma='scale', probability=True, random_state=self.random_state),
            'RandomForestClassifier_3': RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=5, random_state=self.random_state),
            'GradientBoostingClassifier_3': GradientBoostingClassifier(learning_rate=0.05, n_estimators=50, max_depth=3, random_state=self.random_state),
            'AdaBoostClassifier_3': AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME.R', random_state=self.random_state),
            'BaggingClassifier_3': BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=100, random_state=self.random_state),
            'ExtraTreesClassifier_3': ExtraTreesClassifier(n_estimators=50, criterion='entropy', max_depth=3, random_state=self.random_state),
            'VotingClassifier_3': VotingClassifier(estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier()), ('gnb', GaussianNB()), ('svc', SVC(probability=True))]),
            'LGBMClassifier_2': LGBMClassifier(learning_rate=0.1, n_estimators=100, max_depth=5, random_state=self.random_state),
            'XGBClassifier_2': XGBClassifier(learning_rate=0.1, n_estimators=50, max_depth=3, random_state=self.random_state),
            'MLPClassifier_2': MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, activation='relu', solver='adam', random_state=self.random_state),
        }

    def normalize_data(self):
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def test_models(self,cluster, count_feature, len_data, count_k_means, cluster_count, log_file = "result_H1.log"):
        self.normalize_data()
        logging.basicConfig(filename=log_file, level=logging.INFO)
        # logging.info(f"**************************{cluster}**************************")
        # logging.info(f"Len Data : {len_data} and number of features : {count_feature}")
        max_acc = 0
        best_model = ""

        for model_name, model in self.models.items():
            # print(f"Start test {model_name}")
            try:
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    self.data, self.target, test_size=self.test_size, random_state=self.random_state
                )

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                predictions = model.predict(X_test)

                # Evaluate accuracy
                accuracy = accuracy_score(y_test, predictions)
                if accuracy > max_acc :
                    max_acc = accuracy
                    best_model = model_name
                # Log the results
                # log.write(f"{model_name} Accuracy: {accuracy:.4f}\n")
                # print(f"{model_name} Accuracy: {accuracy:.4f}\n")
            except Exception as e:
                # Log the error if an exception occurs
                # log.write(f"Error for {model_name}: {str(e)}\n")
                # print(f"Error for {model_name}: {str(e)}\n")
                print(f"Error in model : {model_name} ")
        if max_acc > 0.6 :
            logging.info(f"best model: {best_model} and the acc: {max_acc:.4f}")
            logging.info(f"Len Data : {len_data} and number of features : {count_feature}, Total cluster : {count_k_means}, cluster number : {cluster_count}")
            logging.info("************************************************************************************************")
