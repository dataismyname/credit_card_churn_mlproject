import sys, os
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.label_encoder_path = os.path.join('artifacts', 'label_encoder.pkl')
        self.feature_names_path = os.path.join('artifacts', 'feature_names.pkl')
        self.correlated_features_file_path = os.path.join('artifacts', 'target_corr_features.pkl')


    def predict(self, features):
        try:
            # Load the serialized objects
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            label_encoder = load_object(file_path=self.label_encoder_path)
            feature_names = load_object(file_path=self.feature_names_path)
            target_corr_features = load_object(file_path=self.correlated_features_file_path)

            # Apply preprocessing
            data_transformed = preprocessor.transform(features)
            df_transformed = pd.DataFrame(data_transformed, columns=feature_names)

            # Selection of highly correlated features 
            df_final = df_transformed[target_corr_features]
            final_input_arr = np.array(df_final)

            # Getting encoded class prediction
            predictions = model.predict(final_input_arr)
            # Convert scalar to 1D array
            predictions = [int(num) for num in predictions]
            # Return original target class value
            decoded_predictions = label_encoder.inverse_transform(predictions)

            return decoded_predictions
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Customer_Age: str,
                Gender: str,
                Dependent_count: str, 
                Education_Level: str,
                Marital_Status: str,
                Income_Category: str,
                Card_Category: str,
                Months_on_book: str,
                Total_Relationship_Count: str,
                Months_Inactive_12_mon: str,
                Contacts_Count_12_mon: str,
                Credit_Limit: str,
                Total_Revolving_Bal: str,
                Avg_Open_To_Buy: str,
                Total_Amt_Chng_Q4_Q1: str,
                Total_Trans_Amt: str,
                Total_Trans_Ct: str,
                Total_Ct_Chng_Q4_Q1: str,
                Avg_Utilization_Ratio: str
                ):
            
        self.Customer_Age = int(Customer_Age)
        self.Gender = Gender
        self.Dependent_count = int(Dependent_count)
        self.Education_Level = Education_Level
        self.Marital_Status = Marital_Status
        self.Income_Category = Income_Category
        self.Card_Category = Card_Category
        self.Months_on_book = int(Months_on_book)
        self.Total_Relationship_Count = int(Total_Relationship_Count)
        self.Months_Inactive_12_mon = int(Months_Inactive_12_mon)
        self.Contacts_Count_12_mon = int(Contacts_Count_12_mon)
        self.Credit_Limit = float(Credit_Limit)
        self.Total_Revolving_Bal = float(Total_Revolving_Bal)
        self.Avg_Open_To_Buy = float(Avg_Open_To_Buy)
        self.Total_Amt_Chng_Q4_Q1 = float(Total_Amt_Chng_Q4_Q1)
        self.Total_Trans_Amt = float(Total_Trans_Amt)
        self.Total_Trans_Ct = int(Total_Trans_Ct)
        self.Total_Ct_Chng_Q4_Q1 = float(Total_Ct_Chng_Q4_Q1)
        self.Avg_Utilization_Ratio = float(Avg_Utilization_Ratio)
        self.calculate_features()
        
    # Calculate of new features
    def calculate_features(self):
        if self.Months_on_book != 0:  # Prevent division by zero.
            self.Products_year:float = self.Total_Relationship_Count / self.Months_on_book * 12
        else:
            self.Products_year:float = 0
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Customer_Age": [self.Customer_Age],
                "Gender": [self.Gender],
                "Dependent_count": [self.Dependent_count],
                "Education_Level": [self.Education_Level],
                "Marital_Status": [self.Marital_Status],
                "Income_Category": [self.Income_Category],
                "Card_Category": [self.Card_Category],
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive_12_mon": [self.Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [self.Contacts_Count_12_mon],
                "Credit_Limit": [self.Credit_Limit],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Avg_Open_To_Buy": [self.Avg_Open_To_Buy],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio],
                "Products_year": [self.Products_year],

            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
             raise CustomException(e,sys)
        
             


    

