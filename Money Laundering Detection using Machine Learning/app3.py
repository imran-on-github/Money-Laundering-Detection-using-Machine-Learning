
# import pickle
# import numpy as np
# from flask import Flask, request, render_template
# from sklearn.preprocessing import StandardScaler
# import warnings
# import pandas as pd
# from tensorflow.keras.models import load_model
# warnings.filterwarnings('ignore')


# app = Flask(__name__)

# # Load the trained model and scaler
# model = load_model('model.h5')  
# # scaler = pickle.load(open('scaler.pkl', 'rb'))
# sc = StandardScaler()

# df = pd.read_csv("data.csv")

# data = df[['type','amount', 'oldbalanceDest','newbalanceDest']]

# sc.fit_transform(data)

# @app.route('/')
# def home():
#  	return render_template('index.html')

# @app.route('/result', methods=['POST'])
# def predict():
#     if request.method == 'POST':
        
#         # Get user input
#         Type = str(request.form['type'])
#         amount = float(request.form['amount'])
#         oldbalanceDest = float(request.form['oldbalanceDest'])
#         newbalanceDest = float(request.form['newbalanceDest'])
#         # Get user input
    
        
#         # Use the pre-defined StandardScaler 'sc' to transform user input data
         
        
#         input_data = np.array([[type, amount, oldbalanceDest, newbalanceDest]])
#         X = sc.transform(input_data)

#             # Use the trained model to make predictions
#         fraud_probability = model.predict(X)
#         threshold = 0.5
#         is_fraud = "Fraud"
#         if fraud_probability > threshold:
#             is_fraud =  "Non-Fraud"
            

#     return render_template('result.html', prediction= fraud_probability)



# if __name__ == '__main__':
#     app.run()
    
    
    

# from flask import Flask, render_template, request
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import get_custom_objects

# app = Flask(__name__)

# # Load your pre-trained model and StandardScaler (replace these with your actual trained model and scaler)
# model = load_model('model.h5')
# scaler = StandardScaler()

# # Function to preprocess user input and make predictions
# def make_predictions(user_input_data, scaler, model):
#     # Preprocess user input (dummy variables for 'type' and scaling)
#     dum = pd.get_dummies(user_input_data['type'])
#     user_input_data = pd.concat([user_input_data, dum], axis=1)
#     user_input_data.drop(['type'], axis=1, inplace=True)
    
#     # Use the pre-defined StandardScaler 'scaler' to transform user input data
#     user_input_data_scaled = scaler.transform(user_input_data)
    
#     # Use the trained model to make predictions
#     fraud_probability = model.predict(user_input_data_scaled)[0, 0]
    
#     # Define a threshold for classification (e.g., 0.5)
#     threshold = 0.5
#     is_fraud = 1 if fraud_probability > threshold else 0
    
#     return fraud_probability, is_fraud

# # Define the routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user input
#         type_input = request.form['type']
#         amount = float(request.form['amount'])
#         oldbalanceDest = float(request.form['oldbalanceDest'])
#         newbalanceDest = float(request.form['newbalanceDest'])

#         # Create a DataFrame with the user input
#         user_input_data = pd.DataFrame({
#             'type': [type_input],
#             'amount': [amount],
#             'oldbalanceDest': [oldbalanceDest],
#             'newbalanceDest': [newbalanceDest]
#             # Add more columns for other features as needed
#         })

#         # Use the fitted StandardScaler 'scaler' to transform user input data
#         user_input_data_scaled = scaler.fit_transform(user_input_data)

#         # Use the trained model to make predictions
#         fraud_probability = model.predict(user_input_data_scaled)

#         # Define a threshold for classification (e.g., 0.5)
#         threshold = 0.5
#         is_fraud = "Fraud" if fraud_probability > threshold else "Non-Fraud"

        
#         # Make predictions
#         fraud_probability, is_fraud = make_predictions(user_input_data, scaler, model)

#         return render_template('result.html', fraud_probability=fraud_probability, is_fraud=is_fraud)

# if __name__ == '__main__':
#     app.run(debug=True)

# import pickle
# import numpy as np
# from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
# import warnings
# import pandas as pd
# from tensorflow.keras.models import load_model
# import os
# import openpyxl
# from require import *

# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# # Load the trained model
# model = load_model('model.h5')
scaler = StandardScaler()

# # Load your dataset
# df = pd.read_csv("data.csv")

# # Remove unwanted features
# df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# # Create dummy variables for categorical values
# dum = pd.get_dummies(df['type'])
# df = pd.concat([df, dum], axis=1)
# df.drop(['type'], axis=1, inplace=True)

# # Splitting the data into training and test
# X_train, X_test, y_train, y_test = train_test_split(df.drop(['isFraud'], axis=1), df['isFraud'], test_size=0.3, random_state=0)

# # Resample the training data
# sm = SMOTE(random_state=10, sampling_strategy=1.0)
# X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# # Feature scaling
# x_train_scaled = scaler.fit_transform(X_train_res)
# x_test_scaled = scaler.transform(X_test)

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from require import *

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


app = Flask(__name__)

# Load your pre-trained model
model = load_model('model.h5')

# Function to make predictions
def make_predictions(user_input_data, sc, model):
    # Preprocess user input (dummy variables for 'type' and scaling)
    dum = pd.get_dummies(user_input_data['type'])
    user_input_data = pd.concat([user_input_data, dum], axis=1)
    user_input_data.drop(['type'], axis=1, inplace=True)
    
    # Use the pre-defined StandardScaler 'sc' to transform user input data
    user_input_data_scaled = x_test_scaled
    # Use the trained model to make predictions
    fraud_probability = model.predict(user_input_data_scaled)[0,0]
    
    # Define a threshold for classification (e.g., 0.5)
    threshold = 0.5
    is_fraud = 1 if fraud_probability > threshold else 0
    
    return fraud_probability, is_fraud



@app.route('/')
def home():
	return render_template('index.html')

# Create a route to display the input form
@app.route('/predict', methods=['GET','POST'])
def prediction():
    fraud_probability=""
    is_fraud =""
    if request.method == 'POST':
        type_input = request.form['tr']
        amount = float(request.form['amount'])
        old_balance_dest = float(request.form['old_balance_dest'])
        new_balance_dest = float(request.form['new_balance_dest'])
        
        # Create a DataFrame with the user input
        user_input_data = pd.DataFrame({
            'type': [type_input],
            'amount': [amount],
            'oldbalanceDest': [old_balance_dest],
            'newbalanceDest': [new_balance_dest],
        })

        fraud_probability, is_fraud = make_predictions(user_input_data, sc, model)
        
    return render_template('result.html', fraud_probability=fraud_probability, is_fraud=is_fraud)

if __name__=='__main__':
    app.run()




