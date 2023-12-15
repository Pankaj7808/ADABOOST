import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Disable warnings (optional)
warnings.filterwarnings("ignore")

data = pd.read_csv("apples_and_oranges.csv")

print(data)

training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
X_train = training_set.iloc[:, 0:2].values
Y_train = training_set.iloc[:, 2].values
X_test = test_set.iloc[:, 0:2].values
Y_test = test_set.iloc[:, 2].values  # Fixed the typo here

adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=1, random_state=1)
adaboost.fit(X_train, Y_train)
Y_pred = adaboost.predict(X_test)
test_set["Predictions"] = Y_pred
print(Y_pred)
