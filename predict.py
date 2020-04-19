import pickle
# import sys

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

#Predict for this vector (height, wieghts, shoe size)
P = [[185, 68, 44]]

# load the model back into memory
f2 = open('models/DecisionTree.pkl', 'rb')
clf2 = pickle.load(f2)

print("\n# Decision Tree Prediction is " + str(clf2.predict(P)))

# load the model back into memory
f2 = open('models/KNeighbors.pkl', 'rb')
knn2 = pickle.load(f2)

print("# K-Neighbors Classifier Prediction is " + str(knn2.predict(P)))

# load the model back into memory
f2 = open('models/RandomForest.pkl', 'rb')
rfor2 = pickle.load(f2)

print("# RandomForestClassifier Prediction is " + str(rfor2.predict(P)) +"\n")
