import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='bpratik01', repo_name='IRIS_REMOTE_SERVER', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/bpratik01/IRIS_REMOTE_SERVER.mlflow')

data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

n_estimators = 100
max_depth = 5

mlflow.set_experiment('iris-rf')

with mlflow.start_run(run_name='random_forest_md=5_ne=100'):
  model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_param('n_estimators', n_estimators)
  mlflow.log_param('max_depth', max_depth)

  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(6,7))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names, cbar=False)
  plt.xlabel('Predicted')
  plt.ylabel('Truth')

  plt.savefig('confusion_matrix_rf.png')

  mlflow.log_artifact('confusion_matrix_rf.png', 'images')

  mlflow.log_artifact('iris_rf.py')

  mlflow.sklearn.log_model(model, 'model')

  print(f'Accuracy: {accuracy}')

  