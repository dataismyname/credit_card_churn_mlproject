import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

"""import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping"""

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split train and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVC": SVC(),
                "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
                "GaussianNB": GaussianNB(),
            }

            hyp_params = {
            "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
            },
            "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_leaf': [1, 2, 4]
            },
            "KNeighbors": {
                    'n_neighbors': [3, 5, 11],
                    'weights': ['uniform', 'distance']
            },
            "Random Forest": {
                    'n_estimators': [10, 50, 100],
                    'max_depth': [10, 20, None],
            },
            "SVC": {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
            },
            "LinearDiscriminantAnalysis": {
                    'solver': ['svd', 'lsqr', 'eigen']
                },
            "GaussianNB": {
                    'var_smoothing': [1e-09, 1e-08, 1e-10]
            }
            }
            
            model_report:dict=evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                             models = models, hyp_params = hyp_params)
            
            # Extract the best model based on Test Score
            best_model, best_model_details = max(model_report.items(), key=lambda x: x[1]["Test Score"])

            best_model_score = best_model_details["Test Score"]
            best_hyp = best_model_details["Best Parameters"]

            if best_model_score<0.75:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            best_model_instance = models[best_model].set_params(**best_hyp)

            # Now, fit the best model instance to your training data
            best_model_instance.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_instance
            )

            # After fitting, you can use this model to predict on X_test
            predictions = best_model_instance.predict(X_test)
            class_values = ['Existing Customer' if pred == 1 else 'Attrited Customer' for pred in predictions]


            acc_score = accuracy_score(y_test, predictions)

            return class_values, acc_score
            
        except Exception as e:
            raise CustomException(e,sys)