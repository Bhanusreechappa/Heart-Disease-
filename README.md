# Heart-Disease-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('C:\\Users\\LENDI\\Downloads\\Heart_Disease_Prediction.csv')
df.head()
df.duplicated().any()
df.plot(kind = "box" , subplots = True , figsize = (14,10) , layout = (5,4))
plt.show()
df['Heart Disease'].value_counts()
df2=df.copy()
mapping = {'Absence': 0, 'Presence': 1}
df2['Heart Disease'] = df2['Heart Disease'].map(mapping)
df2.head()
corr_matrix = df2.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix,cmap='coolwarm', annot=True)
plt.show()
x= df2.groupby('Thallium')['Heart Disease'].sum().index
y=df2.groupby('Thallium')['Heart Disease'].sum()
# to create bar plot with custom colors
colors = sns.color_palette("Set2", len(df2['Thallium'].unique()))
plt.figure(figsize=(10, 6))
sns.barplot(x=x,y=y,data=df2)
plt.title('Heart Disease by Thallium')
plt.ylabel('Heart Disease')
plt.show()
X = df2.drop('Heart Disease', axis=1)
Y = df2['Heart Disease']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, classification_report
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model with  accuracy, F1 score, precision and recall
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
classifcation_report_logistic = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)
print("Classification Report:")
print(classifcation_report_logistic)
# confusion matrix for model performance assessment
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16}, cbar=False, square=True)
# Add labels and titles
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
# Show the plot
plt.show()


