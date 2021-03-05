

import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib

app = Flask(__name__)

model = joblib.load("salary_predictor.pkl")
df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features=[int(x) for x in request.form.values()]
    feature_value=np.array(input_features)
    
    
    
        
    output=model.predict([feature_value])[0][0].round(2)
    output1=output*64
    df=pd.concat([df,pd.DataFrame({'The predicted salary is rs':[output1]})],ignore_index=True)
    print(df)
    df.to_csv('smp_data_from_app.csv')
    
    return render_template('index.html',prediction_text='Your predicted salary is Rs {}'.format(output1))


if __name__=="__main__":
    app.run()
