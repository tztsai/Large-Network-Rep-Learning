from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
# Standarize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
# Create one-vs-rest logistic regression object
clf = LogisticRegression(random_state=0, multi_class='ovr')
# Train model
model = clf.fit(X_std, y)

print(y)
