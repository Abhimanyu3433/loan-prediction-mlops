import joblib
import pandas as pd
 
model = joblib.load("D:\OneDrive - Coforge Limited\Desktop\AI Training\MLOPs Loan Default/model/loan_default.pkl")
 
# create a test sample data
test_df = pd.read_csv("D:\OneDrive - Coforge Limited\Desktop\AI Training\MLOPs Loan Default/data/x_test_sample.csv")
test_df.drop('Unnamed: 0', axis = 1, inplace = True)
 
print(test_df.head())
 
x_test_sample = test_df.sample(1)
 
yhat = model.predict(x_test_sample)
 
print(f"Prediction: {yhat}")