# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from helper_function import AgeFeatureExtract, OutlierClipping, datatype_converter
import pandas as pd
app = Flask(__name__)

filename = 'regressor.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/back')
def back():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('Hey there')
#    input_data = jsonify(request.form.to_dict())
    #print('Ip is this')
    input_data = request.form.to_dict()
    input_df = pd.DataFrame(input_data, index=[0])
    input_df = datatype_converter(input_df)
    prediction = model.predict(input_df)
    pred = np.round(prediction[0], 2)
    input_data['Prediction'] = str(pred)
    #input_data = jsonify(input_data)
    #print(input_data)
    return render_template('result.html', output_data = input_data, pred = pred)
    #return str(prediction[0])

#@app.route('/download', methods=['POST'])
#def download():
#    print('check')
#    pass
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

