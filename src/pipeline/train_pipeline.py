import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

##Main Class Definition: A class that manages the sequential execution of each component in the pipeline
class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

##Method to Execute the Pipeline: This method coordinates the execution of each stage and handles exceptions to ensure smooth execution.
    def run_pipeline(self, df=None):
        try:
            # Data Ingestion
            logging.info("Initiating Data Ingestion.")
            train_data, test_data = self.data_ingestion.initiate_data_ingestion(df=df)

            # Data Transformation
            logging.info("Initiation Data Transformation.")
            train_arr, test_arr = self.data_transformation.initiate_data_transformation(train_data, test_data)

            # Model Training
            logging.info("Initiating Model Training.")
            model, accuracy = self.model_trainer.initiate_model_trainer(train_arr, test_arr)

            logging.info(f"Training Complete. Model Precision: {accuracy:.2f}")

        except Exception as e:
            logging.error(f"Error during pipeline execution: {e}")
            raise CustomException(e, sys)
        
#Script Execution
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()