import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


st.title("GOLD PRICE PREDICTION")

st.header("Yashas E Sankol:1BG21CS117")
st.header("Tejas Nagaraj Hegde:1BG21CS125")
st.header("Srinidhi Vasishta G V:1BG21CS129")
st.image("shutterstock_536623408_5.jpg")
# st.write("This is a simple web app that predicts the gold price based on the data provided by the dataset")
# st.write("The dataset is taken from Kaggle")
gold_data = pd.read_csv('gold_price_data.csv')

gold_data

gold_data.shape



gold_data.info()


gold_data.isnull().sum()


numeric_columns = gold_data.select_dtypes(include=['float64', 'int64']).columns
gold_data_numeric = gold_data[numeric_columns]

correlation = gold_data_numeric.corr()


plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size' : 8}, cmap='Blues')


gold_data.drop(columns=["Date"], axis=1, inplace=True)

# gold_data.head()


X = gold_data.drop(columns=['GLD'], axis=1)
y = gold_data['GLD']

X.head()

y.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)


model = RandomForestRegressor(n_estimators=100)


model.fit(X_train, y_train)


prediction_X_train = model.predict(X_train)

print(f"Accuracy on training data: {r2_score(y_train, prediction_X_train)}")

prediction_X_test = model.predict(X_test)

print(f"Accuracy on testing data: {r2_score(y_test, prediction_X_test)}")



y_test = y_test.reset_index(drop=True)

# for i in range(len(prediction_X_test)):
#   print(f"Predicted Gold Price: ${prediction_X_test[i]}. Actual Gold Price: ${y_test[i]}")
# st.write("Predicted VS Actual Values")
# for i in range(len(prediction_X_test)):
#   st.write(prediction_X_test[i],"\t",y_test[i])

predicted_X_test_plot = list(prediction_X_test)
y_test_plot = list(y_test)

col1, col2, col3 = st.columns(3)

with col1:
    SPX = st.text_input('SPX')
        
with col2:
    GLD = st.text_input('USO')
         
with col3:
    SLV = st.text_input('SLV')
with col1:
    EUR_USD = st.text_input('EUR/USD')
    
outcome = ''

input_data = (SPX,GLD,SLV,EUR_USD) 
input_data_as_numpy_array = np.asarray(input_data) 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 
if st.button("Predict"):
    prediction = model.predict(input_data_reshaped)[0] 
    prediction=prediction/1.97
    st.write(prediction)

plt.plot(predicted_X_test_plot, color='blue', label='Predicted Value')
plt.plot(y_test_plot, color='red', label='Actual Value')
plt.title('Predicted Values VS Actual Values')
plt.xlabel('Number of Values')
plt.ylabel('Gold Price')
plt.legend()
"""### Predicted Value VS Actual Value"""
df= pd.DataFrame(    predicted_X_test_plot,y_test_plot)
st.line_chart(df)




