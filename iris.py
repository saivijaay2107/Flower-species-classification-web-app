import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv(r"C:\Users\GITAA_004\Downloads\Iris.csv",index_col=0)

#df['Species'] = df['Species'].map({'Iris-setose':0,'Iris-versicolor':1,'Iris-virginica':2})

x = df.drop('Species',axis=1)
y = df['Species']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=11)

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

score = accuracy_score(y_test,y_pred)

pickle_out = open('classifier.pkl','wb')
pickle.dump(classifier, pickle_out)
pickle_out.close()