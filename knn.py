from sklearn import datasets
import  numpy as np
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
  
iris=datasets.load_iris()
x=iris.data
Y=iris.target


df=pd.read_csv('data.csv')
 
X=df[['buying','maint','safety']].values
y=df[['class']]
Le=LabelEncoder()
for i in range(len(X[0])):
    X[:,i]=Le.fit_transform(X[:,i])

label_mapping={
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3

}
y['class']=y['class'].map(label_mapping)
y=np.array(y)



knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform')
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2)


knn.fit(x_train,y_train)

prediction=knn.predict(x_test)

accuracy=metrics.accuracy_score(y_test,prediction)

print(prediction,'\n',accuracy)

print(y[20])
print(knn.predict(x)[20])


