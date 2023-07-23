import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = datasets.load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

classifier = SVC(kernel='linear', C=1.0, random_state=42)
classifier.fit(X_train_std, y_train)

y_pred = classifier.predict(X_test_std)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
