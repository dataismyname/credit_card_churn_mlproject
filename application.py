import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
from src.pipeline.predictive_pipeline import PredictPipeline, CustomData
from src.pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)
app = application

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to recieve the data from HTML form or JSON
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    
    try:
        if request.is_json:
            json_data = request.get_json()
            data = CustomData(**json_data)
        else:
            data = CustomData(
                Months_on_book=request.form.get('Months_on_book'),
                Total_Relationship_Count=request.form.get('Total_Relationship_Count'),
                Months_Inactive_12_mon=request.form.get('Months_Inactive_12_mon'),
                Contacts_Count_12_mon=request.form.get('Contacts_Count_12_mon'),
                Total_Revolving_Bal=request.form.get('Total_Revolving_Bal'),
                Total_Amt_Chng_Q4_Q1=request.form.get('Total_Amt_Chng_Q4_Q1'),
                Total_Trans_Amt=request.form.get('Total_Trans_Amt'),
                Total_Trans_Ct=request.form.get('Total_Trans_Ct'),
                Total_Ct_Chng_Q4_Q1=request.form.get('Total_Ct_Chng_Q4_Q1'),
                Avg_Utilization_Ratio=request.form.get('Avg_Utilization_Ratio')
            )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        preds, conf = predict_pipeline.predict(pred_df)

        if request.is_json:
            return jsonify({'prediction': preds[0], 'confidence': conf[0]})

        return render_template('prediction.html', preds=preds[0], conf=conf[0], results=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Upload config
if not os.path.exists('uploads'):
    os.makedirs('uploads')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/retrain', methods=['GET', 'POST'])
def retrain_datapoint():
    if request.method == 'GET':
        return render_template('retrain.html')

    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only CSV files are allowed."}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)

        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to read CSV file: {str(e)}"}), 400

        required_columns = [
            'CLIENTNUM', "Attrition_Flag", "Customer_Age", "Gender", "Dependent_count", "Education_Level",
            "Marital_Status", "Income_Category", "Card_Category", "Months_on_book", "Total_Relationship_Count",
            "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
            "Avg_Open_To_Buy", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
            "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"
        ]

        if not all(col in data.columns for col in required_columns):
            return jsonify({"error": "Missing required columns"}), 400

        pipeline = TrainPipeline()
        pipeline.run_pipeline(df=data)

        return jsonify({"message": "Model retraining completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug = True)