import pickle
import sys

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#{height, weights, shoe size}
# features = [[190,70,44],[166,65,45],[190,90,47],[175,64,39],[171,75,40],[177,80,42],[160,60,38],[144,54,37]]
# labels = ['male','male','male','male','female','female','female','female']

names = ['height', 'weight', 'shoe-size', 'sex']
dataset = pd.read_csv('datasets/dataset.csv', names=names)

array = dataset.values
features = array[:,0:3]
labels = array[:,3]

x_train,x_test,y_train,y_test=train_test_split(features,labels)

#{Decision Tree Model}
clf1 = DecisionTreeClassifier()
clf1 = clf1.fit(x_train,y_train)

print("\n# Decision Tree Accuracy score is " + str(accuracy_score(y_test,clf1.predict(x_test))))

# serialize the model on disk in the special 'outputs' folder
f = open('models/DecisionTree.pkl', 'wb')
pickle.dump(clf1, f)
f.close()

print("# DecisionTreeClassifier model generated!\n")

#{K Neighbors Classifier}
knn = KNeighborsClassifier()
knn.fit(features,labels)

print("\n# K-Neighbors Classifier Accuracy score is " + str(accuracy_score(y_test,knn.predict(x_test))))

# serialize the model on disk in the special 'outputs' folder
f = open('models/KNeighbors.pkl', 'wb')
pickle.dump(knn, f)
f.close()

print("# KNeighborsClassifier model generated!\n")

#{using MLPClassifier}
rfor = RandomForestClassifier()
rfor.fit(features,labels)

print("\n# RandomForestClassifier Accuracy score is " + str(accuracy_score(y_test,rfor.predict(x_test))))

# serialize the model on disk in the special 'outputs' folder
f = open('models/RandomForest.pkl', 'wb')
pickle.dump(rfor, f)
f.close()

print("# RandomForestClassifier model generated!\n")
