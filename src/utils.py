import os
import sys

import dill

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, hyp_params):
    try:
        report = {}
        
        for model_name, model in models.items():
            # Access the corresponding hyperparameters
            hyp_param = hyp_params[model_name]

            # Perform GridSearchCV to find the best hyperparameters
            gs = GridSearchCV(model, hyp_param, cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)

            # The best estimator after GridSearchCV is already fitted with the best parameters
            best_model = gs.best_estimator_
            
            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate scores
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Update the report
            # Now including both train and test scores for more comprehensive reporting
            # Also includes best parameters found by GridSearchCV
            report[model_name] = {
                "Train Score": train_model_score,
                "Test Score": test_model_score,
                "Best Parameters": gs.best_params_
            }
            
        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)