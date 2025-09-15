# **Loan Approval Prediction**
## 1. Setup and Data Loading
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

File_Path = 'loan_approval_dataset.csv'

df = pd.read_csv(File_Path)
df.columns

## 2. Data Exploration and Preprocessing
### 2.1 Data Inspection

df.head()
df.tail()
df.info()
df.describe()

### 2.2 Data Cleaning

df =  df.drop_duplicates()

df = df.dropna(axis = 0)

df.isnull().sum()

### 2.3 Feature Engineering and Encoding

df[' loan_status'].value_counts()

df[' loan_status'].value_counts(normalize=True)

df.select_dtypes(include='object').head()

df.select_dtypes(include=['int64', 'float64']).head()

df = df.drop('loan_id',axis =1 )
# because from id it can learn wrong patterns

df[' loan_status'] = df[' loan_status'].map({' Approved': 1, ' Rejected': 0})
df[' education'] = df[' education'].map({' Graduate': 1, ' Not Graduate': 0})
df[' self_employed'] = df[' self_employed'].map({' Yes': 1, ' No': 0})

df[' loan_status'].unique()

df.head()

x = df.drop(columns=[' loan_status'])
y = df[' loan_status']

## 3. Model Training and Evaluation
### 3.1 Logistic Regression

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

### Evaluation of Logistic Regression

print('\n\nEvaluation of Logistic Regression')
print('Accuracy Score: ', accuracy_score(y_test,y_pred))
print('Precision Score: ', precision_score(y_test,y_pred))
print('Recall Score: ', recall_score(y_test,y_pred))
print('F1 Score: ', f1_score(y_test,y_pred))

print('\nClassification Report\n',classification_report(y_test,y_pred))

plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

### Checking Overfitting and Underfitting

train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)

print('Training Accuracy: ', train_score)
print('Testing Accuracy: ', test_score)

### 3.2 Decision Tree

dc_model = DecisionTreeClassifier(max_depth=5,random_state=42)
dc_model.fit(x_train,y_train)

dc_y_pred = dc_model.predict(x_test)

### Evaluation Of Decision Tree
print('\n\nEvaluation Of Decision Tree')
print('Accuracy Score: ', accuracy_score(y_test,dc_y_pred))
print('Precision Score: ', precision_score(y_test,dc_y_pred))
print('Recall Score: ', recall_score(y_test,dc_y_pred))
print('F1 Score: ', f1_score(y_test,dc_y_pred))

print('\nClassification Report\n',classification_report(y_test,dc_y_pred))

plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, dc_y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Decision Tree Confusion Matrix')
plt.show()

### Checking For Overfitting or Underfitting

dc_train_score = dc_model.score(x_train,y_train)
dc_test_score = dc_model.score(x_test,y_test)

print('Training Accuracy: ', dc_train_score)
print('Testing Accuracy: ', dc_test_score)

### 3.3 Model Comparison
print('\n\nModel Comparison')
print('Logistic Regression Model')
print('Training Accuracy: ', train_score)
print('Testing Accuracy: ', test_score)

print('\nDecision Tree Model')
print('Training Accuracy: ', dc_train_score)
print('Testing Accuracy: ', dc_test_score)