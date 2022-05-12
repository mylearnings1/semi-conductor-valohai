from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy
import joblib
from io import BytesIO
 
app = FastAPI()
 
model_path = 'model_rf.jbl'
loaded_model = None
 
@app.post("{full_path:path}")
async def predict(data: UploadFile = File(...)):
    img = pd.read_csv(BytesIO(await data.read()))

    global loaded_model
    # Check if model is already loaded
 
    if not loaded_model:
        loaded_model = joblib.load(model_path)
 
    # Predict with the model
    prediction = loaded_model.predict(img)
 
    return f'Predicted_Flower: {prediction}'
