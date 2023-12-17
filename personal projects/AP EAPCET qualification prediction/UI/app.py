# import joblib
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn import tree
import joblib
import sklearn
# Load the model
model = joblib.load('D:\\data analytics\\REAL-TIME PROJECTS\\AP EAPCET prediction\\algorithms used\\AP-EAPCET CLASSIFICATION.pkl')

# Ensure it's a scikit-learn model
if isinstance(model, sklearn.base.BaseEstimator):
    # Rest of your Flask code with the model prediction
    # ...

    app = Flask(__name__)



    @app.route('/')
    def man():
        return render_template('home.html')


    @app.route('/predict', methods=['POST'])
    def home():
        data1 = request.form['a']
        data2 = request.form['b']
        data3 = request.form['c']
        districts=["Anantapur","Chittor","East Godavari","Guntur","Krishna","Kurnool","Nellore","Prakasam","Srikakulam","Visakhapatnam","Vizianagaram","West Godavari","YSR Kadapa"]
        region=["Middle Andhra","North Andhra","Rayalaseema","South Andhra"]
        if data1 in districts:
            data1=districts.index(data1)
            
        if data2 in region:
            data2=region.index(data2)
                    
        data1 = int(data1)
        data2 = int(data2)
        data3 = int(data3)
        
        arr = np.array([[data3, data1, data2]])
                    
        # arr = np.array([[900, 1, 2]])
        pred = model.predict(arr)
        output = pred[0] 
        return render_template('after.html', data=output)


if __name__ == "__main__":
    app.run(debug=True)










