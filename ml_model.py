#importing libraries
import pandas as pd
import numpy as np
import pickle

df=pd.read_csv("SUV_Purchase.csv")

#Feature Engineering
df = df.drop('User ID',axis=1)
df = df.drop('Gender',axis=1) #Feature Engineering

#Loading the data
#method 1 - using iloc X->Independent variable Y->dependent var
X = df.iloc[:,:-1].values #2D array
Y = df.iloc[:,-1:].values #2D array

#spliting the data  into training and testing set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Standard Scaling - Normalizing the data - X_train
# Importing StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler
sst = StandardScaler()
X_train=sst.fit_transform(X_train) #nomalizing

#Training the model

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)

#Testing
#Predicting
y_pred = model.predict(sst.transform(X_test))
print(y_pred)

pickle.dump(model,open('model.pkl','wb')) #we are Serializing our model by creating model.pkl and writing into it by 'wb'
model=pickle.load(open('model.pkl','rb')) #Deserializing - reading the file - "rb"
print("Sucess loaded")


#Execute this file only once and create the pkl file.