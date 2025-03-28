import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats.mstats import winsorize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_file_path: str = os.path.join('artifacts', 'label_encoder.pkl')
    feature_names_file_path: str = os.path.join('artifacts', 'feature_names.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def identify_columns(self, df, exclude_cols=None):
        if exclude_cols is None:
            exclude_cols = []
        num_cols = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]
        cat_cols = [col for col in df.select_dtypes(exclude='number').columns if col not in exclude_cols]
        return num_cols, cat_cols

    def handle_outliers(self, df, skew_threshold=1, limits=(0.05, 0.25)):
        for col in df.select_dtypes(include='number').columns:
            skewness = skew(df[col])
            adjusted_limits = limits if skewness > skew_threshold else (limits[1], limits[0]) if skewness < -skew_threshold else (0.15, 0.15)
            df[col] = winsorize(df[col], limits=adjusted_limits)
        return df

    def create_new_features(self, df, feature_formulas):
        for new_feature, formula in feature_formulas.items():
            df[new_feature] = eval(formula)
        return df

    def save_multiple_objects(self, file_paths, objects):
        for file_path, obj in zip(file_paths, objects):
            save_object(file_path, obj)

    def get_data_transformation_obj(self, features):
        """
        This function sets up pipelines for scaling numeric columns and encoding categorical columns.
        """
        try:
            # Identifying numerical and categorical columns
            num_columns, cat_columns = self.identify_columns(features, exclude_cols=['Attrition_Flag'])

            # Pipelines and Column Transformer
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            cat_pipeline = Pipeline(steps=[("ohe", OneHotEncoder())])

            logging.info(f"Numerical columns: {num_columns}")
            logging.info(f"Categorical columns: {cat_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns),
                ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        selected_features = [
            'Months_on_book',
            'Total_Relationship_Count',
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon',
            'Total_Revolving_Bal',
            'Total_Amt_Chng_Q4_Q1',
            'Total_Trans_Amt',
            'Total_Trans_Ct',
            'Total_Ct_Chng_Q4_Q1',
            'Avg_Utilization_Ratio',
            'Attrition_Flag'
        ]

        try:
            # Reading the data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")
            train_df = train_df[selected_features]
            test_df = test_df[selected_features]

            # Handling outliers
            train_df = self.handle_outliers(train_df)
            test_df = self.handle_outliers(test_df)

            # Creating new features
            feature_formulas = {
                'Products_year': 'df["Total_Relationship_Count"] / df["Months_on_book"] * 12'
            }
            train_df = self.create_new_features(train_df, feature_formulas)
            test_df = self.create_new_features(test_df, feature_formulas)

            # Defining target column
            target_column = 'Attrition_Flag'
            train_features = train_df.drop(columns=[target_column], axis=1)
            test_features = test_df.drop(columns=[target_column], axis=1)

            # Getting the preprocessing object
            preprocessing_obj = self.get_data_transformation_obj(train_features)

            # Applying transformations
            train_features_transformed = preprocessing_obj.fit_transform(train_features)
            test_features_transformed = preprocessing_obj.transform(test_features)

            # Saving the preprocessing object
            enc_feat_col = pd.get_dummies(train_features).columns

            le = LabelEncoder()
            train_target_encoded = le.fit_transform(train_df[target_column])
            test_target_encoded = le.transform(test_df[target_column])

            # Saving multiple objects
            objects_to_save = [
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path,
                self.data_transformation_config.feature_names_file_path
            ]
            objects = [preprocessing_obj, le, enc_feat_col]
            self.save_multiple_objects(objects_to_save, objects)

            # Combining transformed features and encoded target
            final_columns = list(enc_feat_col) + [target_column]
            train_arr = np.column_stack((train_features_transformed, train_target_encoded))
            test_arr = np.column_stack((test_features_transformed, test_target_encoded))

            return train_arr, test_arr

        except Exception as e:
            error_message = f"Error occurred: {str(e)}"
            logging.error(error_message)
            raise CustomException(error_message, sys)