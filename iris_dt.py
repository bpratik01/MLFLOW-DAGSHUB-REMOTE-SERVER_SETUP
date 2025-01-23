import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import dagshub
dagshub.init(repo_owner='bpratik01', repo_name='IRIS_REMOTE_SERVER', mlflow=True)


mlflow.set_tracking_uri('https://dagshub.com/bpratik01/IRIS_REMOTE_SERVER.mlflow')
data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

max_depth = 5

mlflow.set_experiment('iris-dt')

with mlflow.start_run(run_name = 'detree_iris_md=5'):
    model = DecisionTreeClassifier(max_depth=max_depth)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png', 'images')

    mlflow.log_artifact('iris_dt.py')

    mlflow.sklearn.log_model(model, 'model')

    print(f'Accuracy: {accuracy}')

    