import pandas as pd
import numpy as np

# Extract features and predictors
data = pd.read_csv('parkinsons.data')

predictors = data.drop(['name','status'], axis=1).to_numpy()
target = data['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
X = scaler.fit_transform(predictors)
Y = target

# Split training data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

# Create KNN model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("k-Nearest Neighbor:")
print(metrics.accuracy_score(Y_test, y_pred))
print(metrics.classification_report(Y_test, y_pred))
print(metrics.confusion_matrix(Y_test, y_pred))