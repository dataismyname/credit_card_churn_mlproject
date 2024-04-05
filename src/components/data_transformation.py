import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        """
        This function is responsible of data transformation
        """
        try:
            num_columns = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                            'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                            'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
            
            cat_columns = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 
                           'Card_Category']
            target_column = ['Attrition_Flag']
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("ohe", OneHotEncoder()),
                ]
            )

            logging.info(f"Numerical columns: {num_columns}")
            logging.info(f"Categorical columns: {cat_columns}")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, num_columns),
                    ("cat_pipeline", cat_pipeline, cat_columns),
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformation_obj()

            target_column_name = 'Attrition_Flag'
            
            le = LabelEncoder()
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = le.fit_transform(train_df[target_column_name])
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = le.transform(test_df[target_column_name])

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        