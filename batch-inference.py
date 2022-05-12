import json
from zipfile import ZipFile

import pandas as pd
import tensorflow as tf
import valohai 
import joblib

model_path1 = 'model_rf.jbl'
loaded_model1 = None
model_path2 = 'fatures_selected.jbl'
loaded_model2 = None

if not loaded_model1:
    output_path = valohai.inputs('model1').path('model_rf.jbl')
    with open(output_path,'rb') as f:
        loaded_model1 = joblib.load(f)


inp = valohai.inputs('images').path()
csv = pd.read_csv(inp)
csv = csv.drop(columns = ['Time'], axis = 1)
labels = csv.pop('Pass/Fail')

if not loaded_model2:
    output_path = valohai.inputs('model2').path('fatures_selected.jbl')
    with open(output_path,'rb') as f:
        loaded_model2 = joblib.load(f)

csv = pd.DataFrame(csv, columns=loaded_model2)

data = tf.data.Dataset.from_tensor_slices((dict(csv), labels))
batch_data = data.batch(batch_size=10)

results = loaded_model1.predict_batch(batch_data)

# Let's build a dictionary out of the results,
# e.g. {"1": 0.375, "2": 0.76}
flattened_results = results.flatten()
indexed_results = enumerate(flattened_results, start=1)
metadata = dict(indexed_results)

for value in metadata.values():
    with valohai.logger() as logger:
        logger.log("result", value)

with open(valohai.outputs().path('results.json'), 'w') as f:
    # The JSON library doesn't know how to print
    # NumPy float32 values, so we stringify them
    json.dump(metadata, f, default=lambda v: str(v))
