from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import pickle



app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
model = pickle.load(open('LRModel.pkl','rb'))

@app.route('/')
def index():
    
    locations = sorted(data['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = request.form.get('sqft')
    total_sqft = request.form.get('sqft')
    print(location,bhk,bath,sqft,total_sqft)
    
    input = pd.DataFrame(
        [[location,bhk,bath,sqft,total_sqft]],
        columns=['location','bhk','bath','sqft','total_sqft']
    )
    prediction = abs(model.predict(input)[0] * 1e5)
    return str(np.round(prediction,2))


if __name__ == '__main__':
    app.run(debug=True,port=5001)