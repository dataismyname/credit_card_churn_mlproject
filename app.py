from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predictive_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

#Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Customer_Age = request.form.get('Customer_Age'),
            Gender = request.form.get('Gender'),
            Dependent_count = request.form.get('Dependent_count'),
            Education_Level = request.form.get('Education_Level'),
            Marital_Status = request.form.get('Marital_Status'),
            Income_Category = request.form.get('Income_Category'),
            Card_Category = request.form.get('Card_Category'),
            Months_on_book = request.form.get('Months_on_book'),
            Total_Relationship_Count = request.form.get('Total_Relationship_Count'),
            Months_Inactive_12_mon = request.form.get('Months_Inactive_12_mon'),
            Contacts_Count_12_mon = request.form.get('Contacts_Count_12_mon'),
            Credit_Limit = request.form.get('Credit_Limit'),
            Total_Revolving_Bal = request.form.get('Total_Revolving_Bal'),
            Avg_Open_To_Buy = request.form.get('Avg_Open_To_Buy'),
            Total_Amt_Chng_Q4_Q1 = request.form.get('Total_Amt_Chng_Q4_Q1'),
            Total_Trans_Amt = request.form.get('Total_Trans_Amt'),
            Total_Trans_Ct = request.form.get('Total_Trans_Ct'),
            Total_Ct_Chng_Q4_Q1 = request.form.get('Total_Ct_Chng_Q4_Q1'),
            Avg_Utilization_Ratio = request.form.get('Avg_Utilization_Ratio')
            )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        decoded_predictions=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=decoded_predictions[0])
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", debug = True)
    

