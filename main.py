import mglearn as mglearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.neighbors import KDTree, KNeighborsRegressor


#using pandas to get mean median mode of each class(Gender)
employee = pd.read_csv('Social_Network_Ads.csv')
dept_gender_salary =  employee.groupby(['Gender'],as_index=False).EstimatedSalary.mean()
print("\nAverage Salary based on gender\n")
print(dept_gender_salary)
dept_gender_age =  employee.groupby(['Gender'],as_index=False).Age.mean()
print("\nAverage Age based on gender\n")
print(dept_gender_age)
dept_gender_age_median = employee.groupby(['Gender'],as_index=False).Age.median()
print("\nMedian Age based on gender\n")
print(dept_gender_age_median)
dept_gender_age_median_gender = employee.groupby(['Gender'],as_index=False).EstimatedSalary.median()
print("\nMedian Salary based on gender\n")
print(dept_gender_age_median_gender)
dept_gender_age_mode = employee.groupby(['Gender'],as_index=False).Age.describe()
print(dept_gender_age_mode )
dept_gender_age_mode_gender = employee.groupby(['Gender'],as_index=False).EstimatedSalary.describe()
print(dept_gender_age_mode_gender)

#Importing The Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [0,1, 2, 3]].values
Z = dataset.iloc[:, 2].values
R = dataset.iloc[:, 3].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

print(X[:, 1])
print(Z)
# plot dataset
mglearn.discrete_scatter(X[:, 2], R, y)
plt.legend(["Female", "Male"], loc=4)
plt.xlabel("Age")
plt.ylabel("Salary (USD)")
plt.show()
print("X.shape: {}".format(X.shape))


#plt.scatter(Z, R)
#plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])
print(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state = 0)
print("Training data set")
print(X_train)
print("Testing data set")
print(X_test)
print("Y Train data set")
print(y_train)
print("Y test data set")
print(y_test)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print("this is testing data result")
print(y_test)
print("this is expected data result")
print(y_pred)

#print accuracy and confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
print(cm)
print(ac)

#Kd- tree code
rng = np.random.RandomState(0)
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print("the three indices for the test data ")
print(ind)
print("the distance of the test data")
print(dist)

#trying to print a graph between accuracy and n each time using a graph
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

train_accuracy[i] = knn.score(X_train, y_train)
test_accuracy[i] = knn.score(X_test, y_test)
plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


#Regression coefficient calculation:
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(X_train,y_train)
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


