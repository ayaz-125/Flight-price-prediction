import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

data = pd.read_csv('Deploy_data_pdf')
# print(data)
# print("The names of the columns in the dataset is this",data.columns)
print(data.shape)
data.drop('Unnamed: 0',axis=1,inplace=True)

x = data.drop('Price',axis=1)
y = data['Price']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12,test_size=0.20)

cat_model = CatBoostRegressor()
cat_model.fit(x_train,y_train)
cat_predict = cat_model.predict(x_test)
# print("The prediction with catboost model is ",cat_predict)

from sklearn.metrics import r2_score
acc_cat = r2_score(y_test,cat_predict)
print('The accuracy of the model is ',acc_cat*100)

import pickle
pickle.dump(cat_model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))