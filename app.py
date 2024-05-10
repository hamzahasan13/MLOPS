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
            
            RiskLevel = request.form.get("RiskLevel"),
            fuelType = request.form.get("fuelType"),
            vehicleType = request.form.get("vehicleType"),
            gearbox = request.form.get("gearbox"),
            HorsePower = float(request.form.get("HorsePower")),
            kilometer = float(request.form.get("kilometer")),
            Seller = request.form.get("Seller"),
            NotRepairedDamaged = request.form.get("NotRepairedDamaged"),
            abtest = request.form.get("abtest"),
            offerType = request.form.get("offerType")
            
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return (render_template('home.html', results = results))
    
    
if __name__ == "__main__":
    app.run(host = "0.0.0.0", debug=True)