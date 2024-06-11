import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN

titanic = pd.read_csv("titanic.csv")
titanic = titanic[titanic["Age"].notna()]
# Titanic
svm_titanic = SVC(kernel="linear")


x_t = titanic.drop(["PassengerId", "Name", "Survived", "Ticket", "Cabin", "Fare"], axis=1)
label_encoder = LabelEncoder()
x_t["Sex"] = label_encoder.fit_transform(x_t["Sex"])
x_t["Embarked"] = label_encoder.fit_transform(x_t["Embarked"])

y_t = titanic["Survived"]

adasyn_ = ADASYN(random_state=13)

x ,y = adasyn_.fit_resample(x_t, y_t)

X_train, X_test, y_train, y_test = train_test_split(x_t, y_t, shuffle=True, random_state=13)

svm_titanic.fit(X_train, y_train)

y_pre = svm_titanic.predict(X_test)

print("Accuracy:")
print(accuracy_score(y_test, y_pre))

print("Clf Report:")
print(classification_report(y_test, y_pre))

conf_matrix = confusion_matrix(y_test, y_pre)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=[0, 1],
            yticklabels=[0, 1])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title("Titanic - Confusion Matrix")
plt.show()