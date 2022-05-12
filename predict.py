from fastapi import FastAPI, File, UploadFile
import pandas as pd
import numpy
import joblib
from io import BytesIO
 
app = FastAPI()
 
model_path1 = 'model_rf.jbl'
loaded_model1 = None
model_path2 = 'fatures_selected.jbl'
loaded_model2 = None

@app.post("{full_path:path}")
async def predict(data: UploadFile = File(...)):
    img = pd.read_csv(BytesIO(await data.read()))
    img = img.drop(columns = ['Time'], axis = 1)
    global loaded_model2
    
    loaded_model2 = joblib.load(model_path2)
    img = pd.DataFrame(img, columns=loaded_model2)

    global loaded_model1
    # Check if model is already loaded
 
    if not loaded_model1:
        loaded_model1 = joblib.load(model_path1)
 
    # Predict with the model
    prediction = loaded_model1.predict(img)
 
    return f'Predicted_Value: {prediction}'
