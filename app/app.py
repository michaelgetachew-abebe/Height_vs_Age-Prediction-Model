from flask import Flask
import numpy as np
from joblib import load

app = Flask(__name__)

@app.route('/')
def hello_world():
    test_np_input = np.array([[1], [2], [17]])
    with open('./model.joblib', 'rb') as read_in:
        model = load(read_in)

    predictions = model.predict(test_np_input)
    pred_as_str = str(predictions)

    return pred_as_str