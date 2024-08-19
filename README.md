# SpaceX-Rocket-Landing-Success-Prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
def plot_confusion_matrix(y,y_predict):
"this function plots the confusion matrix"
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_predict)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
plt.show()
FlightNumber Date BoosterVersion PayloadMass Orbit LaunchSite Outcome Flights GridFins Reused Legs LandingPad Block Reuse
0 1
2010-
06-04 Falcon 9 6123.6 LEO CCSFS SLC
40
None
None 1 False False False NaN 1.0
1 2
2012-
05-22 Falcon 9 525.0 LEO CCSFS SLC
40
None
None 1 False False False NaN 1.0
2 3
2013-
03-01 Falcon 9 677.0 ISS CCSFS SLC
40
None
None 1 False False False NaN 1.0
3 4
2013-
09-29 Falcon 9 500.0 PO VAFB SLC
4E
False
Ocean 1 False False False NaN 1.0
4 5
2013-
12-03 Falcon 9 3170.0 GTO
CCSFS SLC
40
None
None 1 False False False NaN 1.0
data = pd.read_csv('spacex_dataset_part_2.csv')
data.head()
FlightNumber PayloadMass Flights GridFins Reused Legs Block ReusedCount Orbit_ESL1 Orbit_GEO ... Serial_B1048 Serial_B1049
0 1.0 6123.6 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ... 0.0 0.0
1 2.0 525.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ... 0.0 0.0
2 3.0 677.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ... 0.0 0.0
3 4.0 500.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ... 0.0 0.0
4 5.0 3170.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 ... 0.0 0.0
... ... ... ... ... ... ... ... ... ... ... ... ... ...
85 86.0 15600.0 2.0 1.0 1.0 1.0 5.0 12.0 0.0 0.0 ... 0.0 0.0
86 87.0 15600.0 3.0 1.0 1.0 1.0 5.0 13.0 0.0 0.0 ... 0.0 0.0
87 88.0 15600.0 6.0 1.0 1.0 1.0 5.0 12.0 0.0 0.0 ... 0.0 0.0
88 89.0 15600.0 3.0 1.0 1.0 1.0 5.0 12.0 0.0 0.0 ... 0.0 0.0
89 90.0 3681.0 1.0 1.0 0.0 1.0 5.0 8.0 0.0 0.0 ... 0.0 0.0
90 rows × 80 columns
X = pd.read_csv('spacex_dataset_part_3.csv')
X.head(100)
Y = data['Class'].to_numpy()
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
Y_test.shape
(18,)
✂ Logistic Regression
Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the
dictionary parameters.
# Define logistic regression model
lr = LogisticRegression()
# Define parameters to tune
parameters = {'C': [0.01, 0.1, 1],
'penalty': ['l2'],
'solver': ['lbfgs']}
# Create GridSearchCV object
logreg_cv = GridSearchCV(lr, parameters, cv=10)
# Fit the GridSearchCV object
logreg_cv.fit(X_train, Y_train)
# Output the GridSearchCV object for logistic regression
print("Tuned hyperparameters (best parameters):", logreg_cv.best_params_)
print("Accuracy:", logreg_cv.best_score_)
Tuned hyperparameters (best parameters): {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
Accuracy: 0.8214285714285714
# Calculate accuracy on the test data
accuracy_logreg = logreg_cv.score(X_test, Y_test)
print("Accuracy on test data (Logistic Regression):", accuracy_logreg)
Accuracy on test data (Logistic Regression): 0.8333333333333334
yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
 
Examining the confusion matrix, we see that logistic regression can distinguish between the different classes. We see that the major problem is
false positives.
✂ SVM
# Define parameters for grid search
parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
'C': np.logspace(-3, 3, 5),
'gamma':np.logspace(-3, 3, 5)}
# Create SVM object
svm = SVC()
# Create GridSearchCV object
svm_cv = GridSearchCV(svm, parameters, cv=10)
# Fit the GridSearchCV object to find the best parameters
svm_cv.fit(X_train, Y_train)
# Output the best parameters and best score
print("tuned hyperparameters :(best parameters) ", svm_cv.best_params_)
print("accuracy :", svm_cv.best_score_)
tuned hyperparameters :(best parameters) {'C': 1.0, 'gamma': 0.03162277660168379, 'kernel': 'sigmoid'}
accuracy : 0.8482142857142858
# Calculate the accuracy on the test data
accuracy_svm = svm_cv.score(X_test, Y_test)
print("Accuracy on test data (Support Vector Machine):", accuracy_svm)
Accuracy on test data (Support Vector Machine): 0.8333333333333334
yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
 
✂ Decision Tree
# Create the decision tree classifier object
tree = DecisionTreeClassifier()
# Define the parameters for hyperparameter tuning
parameters = {
'criterion': ['gini', 'entropy'],
'splitter': ['best', 'random'],
'max_depth': [2*n for n in range(1,10)],
'max_features': [None, 'sqrt', 'log2'],
'min_samples_leaf': [1, 2, 4],
'min_samples_split': [2, 5, 10]
}
# Create a GridSearchCV object
tree_cv = GridSearchCV(tree, parameters, cv=10)
# Fit the GridSearchCV object to find the best parameters
tree_cv.fit(X_train, Y_train)
# Output the best parameters and accuracy
print("Tuned hyperparameters (best parameters):", tree_cv.best_params_)
print("Accuracy:", tree_cv.best_score_)
 
Tuned hyperparameters (best parameters): {'criterion': 'gini', 'max_depth': 6, 'max_features': None, 'min_samples_leaf': 4, 'min_samples
Accuracy: 0.8857142857142858
# Calculate the accuracy on the test data using the method score
accuracy_tree = tree_cv.score(X_test, Y_test)
print("Test Accuracy (Decision Tree):", accuracy_tree)
Test Accuracy (Decision Tree): 0.7777777777777778
yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
 
✂ KNN
# Define the parameters for grid search
parameters = {
'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
'p': [1, 2]
}
# Create a KNN classifier object
KNN = KNeighborsClassifier()
# Create a GridSearchCV object
knn_cv = GridSearchCV(KNN, parameters, cv=10)
# Fit the GridSearchCV object to find the best parameters
knn_cv.fit(X_train, Y_train)
# Output the best parameters and best score
print("Tuned hyperparameters (best parameters):", knn_cv.best_params_)
print("Best accuracy:", knn_cv.best_score_)
Tuned hyperparameters (best parameters): {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1}
Best accuracy: 0.8339285714285714
# Calculate the accuracy of knn_cv on the test data
accuracy_knn = knn_cv.score(X_test, Y_test)
print("Accuracy on test data (K Nearest Neighbors):", accuracy_knn)
Accuracy on test data (K Nearest Neighbors): 0.7777777777777778
yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)
Results
Logistic Regression:

Best Hyperparameters: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
Accuracy on Test Data: 0.833
Support Vector Machine (SVM):

Best Hyperparameters: {'C': 1.0, 'gamma': 0.03162277660168379, 'kernel': 'sigmoid'}
Accuracy on Test Data: 0.833
Decision Tree Classifier:

Best Hyperparameters: {'criterion': 'gini', 'max_depth': 6, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 2}
Accuracy on Test Data: 0.778
K-Nearest Neighbors (KNN):

Best Hyperparameters: {'algorithm': 'auto', 'n_neighbors': 3, 'p': 1}
Accuracy on Test Data: 0.778
