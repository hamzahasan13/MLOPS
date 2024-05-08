from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for homepage

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(
            HorsePower = request.form.get("HorsePower"),
            kilometer = request.form.get("kilometer"),
            Risk_Level_Low = request.form.get("Risk_Level_Low"),
            Risk_Level_High = request.form.get("Risk_Level_High"),
            fuelType_Diesel = request.form.get("fuelType_Diesel"),
            vehicleType_Convertible = request.form.get("vehicleType_Convertible"),
            gearbox_Automatic = request.form.get("gearbox_Automatic"),
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return (render_template('home.html', results = results[0]))
    
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug=True)