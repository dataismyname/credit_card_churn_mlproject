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

            # Apply preprocessing
            data_transformed = preprocessor.transform(features)
            df_transformed = pd.DataFrame(data_transformed, columns=feature_names)

            # Selection of highly correlated features 
            #df_final = df_transformed[feature_names]
            final_input_arr = np.array(df_transformed)

            # Getting encoded class prediction
            predictions = model.predict(final_input_arr)
            
            # Convert scalar to 1D array
            predictions = [int(num) for num in predictions]
            
            # Return original target class value
            decoded_predictions = label_encoder.inverse_transform(predictions)

            # Obtaining confidence
            probas = model.predict_proba(final_input_arr)

            # Obtaining the associated probability to the predicted class
            prediction_confidence = [round(proba[pred] * 100, 2) for proba, pred in zip(probas, predictions)]

            return decoded_predictions, prediction_confidence

            #return decoded_predictions
        
        except Exception as e:
            raise CustomException(e, sys) 

class CustomData:
    def __init__(
        self,
        Months_on_book: str,
        Total_Relationship_Count: str,
        Months_Inactive_12_mon: str,
        Contacts_Count_12_mon: str,
        Total_Revolving_Bal: str,
        Total_Amt_Chng_Q4_Q1: str,
        Total_Trans_Amt: str,
        Total_Trans_Ct: str,
        Total_Ct_Chng_Q4_Q1: str,
        Avg_Utilization_Ratio: str
    ):
        self.Months_on_book = float(Months_on_book)
        self.Total_Relationship_Count = float(Total_Relationship_Count)
        self.Months_Inactive_12_mon = float(Months_Inactive_12_mon)
        self.Contacts_Count_12_mon = float(Contacts_Count_12_mon)
        self.Total_Revolving_Bal = float(Total_Revolving_Bal)
        self.Total_Amt_Chng_Q4_Q1 = float(Total_Amt_Chng_Q4_Q1)
        self.Total_Trans_Amt = float(Total_Trans_Amt)
        self.Total_Trans_Ct = float(Total_Trans_Ct)
        self.Total_Ct_Chng_Q4_Q1 = float(Total_Ct_Chng_Q4_Q1)
        self.Avg_Utilization_Ratio = float(Avg_Utilization_Ratio)

        # Calculating derivated features
        self.calculate_features()

    def calculate_features(self):
        # Products_year: Number of products acquired in the last year (12 months)
        if self.Months_on_book != 0:
            self.Products_year = (
                self.Total_Relationship_Count / self.Months_on_book * 12
            )
        else:
            self.Products_year = 0

    def get_data_as_dataframe(self):
        """
        Biuld a DataFrame with the columns expected by the pipeline
        """
        try:
            custom_data_input_dict = {
                "Months_on_book": [self.Months_on_book],
                "Total_Relationship_Count": [self.Total_Relationship_Count],
                "Months_Inactive_12_mon": [self.Months_Inactive_12_mon],
                "Contacts_Count_12_mon": [self.Contacts_Count_12_mon],
                "Total_Revolving_Bal": [self.Total_Revolving_Bal],
                "Total_Amt_Chng_Q4_Q1": [self.Total_Amt_Chng_Q4_Q1],
                "Total_Trans_Amt": [self.Total_Trans_Amt],
                "Total_Trans_Ct": [self.Total_Trans_Ct],
                "Total_Ct_Chng_Q4_Q1": [self.Total_Ct_Chng_Q4_Q1],
                "Avg_Utilization_Ratio": [self.Avg_Utilization_Ratio],
                
                # Derivated columns
                "Products_year": [self.Products_year]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
