import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

max_depth = 10

mlflow.set_experiment('iris-dt')

with mlflow.start_run(run_name = 'detree_iris'):
    model = DecisionTreeClassifier(max_depth=max_depth)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    