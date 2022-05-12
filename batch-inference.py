import json
from zipfile import ZipFile

import pandas as pd
import tensorflow as tf
import valohai as vh
import joblib
import sklearn

with open(vh.inputs('model1').path(), 'r') as f:
    model = joblib.load(f)
inp = valohai.inputs('images').path()
csv = pd.read_csv(inp)
csv = csv.drop(columns = ['Time'], axis = 1)
labels = csv.pop('Pass/Fail')
with open(vh.inputs('model2').path(), 'r') as g:
    features = joblib.load(g)
csv= pd.DataFrame(csv, columns=features)

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
