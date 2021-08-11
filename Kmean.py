from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd
bc=load_breast_cancer()
X=scale(bc.data)
y=bc.target
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=6)
model=KMeans(n_clusters=2,random_state=0)
model.fit(x_train)
preditions=model.predict(x_test)
labels=model.labels_
print('labels',labels)
print('predictions',preditions)
print('accuracy',accuracy_score(y_test,preditions))