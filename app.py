# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
#from helper_function import AgeFeatureExtract, OutlierClipping, datatype_converter
import pandas as pd
app = Flask(__name__)

from sklearn.base import BaseEstimator, TransformerMixin

def datatype_converter(df):
    df = df[['dayhours', 'room_bed', 'room_bath', 'living_measure', 'ceil', 'coast',
           'sight', 'condition', 'quality', 'basement', 'yr_built', 'yr_renovated',
           'zipcode', 'living_measure15', 'lot_measure15', 'furnished',
           'total_area']]
    int_columns = [ 'room_bed', 'living_measure', 'coast',
           'sight', 'condition', 'quality', 'basement', 'yr_built', 'yr_renovated',
           'zipcode', 'living_measure15', 'lot_measure15', 'furnished',
           'total_area']
    float_columns = [ 'room_bath', 'ceil']
    df[int_columns] = df[int_columns].astype('int')
    df[float_columns] = df[float_columns].astype('float')
    return df

class AgeFeatureExtract(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  def fit(self, x, y=None):
    return self
  def transform(self, df, y=None):
    sold_year = df.loc[:,'dayhours'].str.extract(r'(\d{4})', expand = False)
    df['house_age'] = sold_year.astype('int64') - df['yr_built']
    df['renovated_age'] = df.apply(lambda x: x['yr_renovated']-x['yr_built'] if x['yr_renovated'] != 0 else 0, axis = 1)
    return df[['house_age', 'renovated_age']]


class OutlierClipping(BaseEstimator, TransformerMixin):
  def __init__(self):
    pass
  def fit(self, df, y=None):
    return self
  def transform(self, df, y=None):
    df.loc[:,'room_bed'] = df.loc[:,'room_bed'].clip(upper=6)
    df.loc[:,'room_bath'] = df.loc[:,'room_bath'].clip(upper=3.75)
    df.loc[:,'ceil'] = df.loc[:,'ceil'].clip(upper=2.5)
    df.loc[:,'condition'] = df.loc[:,'condition'].clip(lower=2)
    df.loc[:,'quality'] = df.loc[:,'quality'].clip(lower=5, upper=11)
    return df[['room_bed', 'room_bath', 'ceil']]

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
