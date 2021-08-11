from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
iris=datasets.load_iris()

X=iris.data
y=iris.target

classes=['Iris Setosa','Iris Versicolour','Iris Verginica']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


model=svm.SVC()
model.fit(X_train,y_train)


prediction=model.predict(X_test)
acc=accuracy_score(y_test,prediction)


print(acc)


for i in range(len(prediction)):
    print(classes[prediction[i]])

                         
