import json
from zipfile import ZipFile

import pandas as pd
import tensorflow as tf
import valohai as vh
import joblib

with open(vh.inputs('model').path('model_rf.jbl'), 'r') as f:
    model = joblib.load(f)

csv = pd.read_csv('data.csv')
labels = csv.pop('Pass/Fail')
data = tf.data.Dataset.from_tensor_slices((dict(csv), labels))
batch_data = data.batch(batch_size=10)

results = model.predict(batch_data)

# Let's build a dictionary out of the results,
# e.g. {"1": 0.375, "2": 0.76}
flattened_results = results.flatten()
indexed_results = enumerate(flattened_results, start=1)
metadata = dict(indexed_results)

for value in metadata.values():
    with vh.logger() as logger:
        logger.log("result", value)

with open(vh.outputs().path('results.json'), 'w') as f:
    # The JSON library doesn't know how to print
    # NumPy float32 values, so we stringify them
    json.dump(metadata, f, default=lambda v: str(v))
