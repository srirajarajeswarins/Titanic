#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("titanic.csv")
df.head()


# In[10]:


# Check for missing values
df.isnull().sum()

# Fill missing values for 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values for 'Embarked' with the most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing values for 'Cabin' with 'Unknown'
df['Cabin'].fillna('Unknown', inplace=True)

# Drop 'Cabin' as it has too many unique values to be useful in this simple model
df.drop(columns=['Cabin'], inplace=True)

# Check again for missing values
df.isnull().sum()


# In[11]:


# Convert 'Sex' into a binary variable: 0 for female, 1 for male
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[12]:


# Create a new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop columns that won't be used in the model
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Check the dataframe after feature engineering
df.head()


# In[13]:


from sklearn.model_selection import train_test_split

# Define features and target
X = df.drop(columns=['Survived'])
y = df['Survived']

# Split the data into training (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[21]:


pip install catboost


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
grad_boost = GradientBoostingClassifier(n_estimators=100, random_state=42)
cat_boost = CatBoostClassifier(iterations=100, random_seed=42, silent=True)

# Train the models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
grad_boost.fit(X_train, y_train)
cat_boost.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_random_forest = random_forest.predict(X_test)
y_pred_grad_boost = grad_boost.predict(X_test)
y_pred_cat_boost = cat_boost.predict(X_test)

# Calculate accuracies
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
accuracy_grad_boost = accuracy_score(y_test, y_pred_grad_boost)
accuracy_cat_boost = accuracy_score(y_test, y_pred_cat_boost)

# Print the accuracies
print(f"Logistic Regression Accuracy: {accuracy_log_reg}")
print(f"Random Forest Accuracy: {accuracy_random_forest}")
print(f"Gradient Boosting Accuracy: {accuracy_grad_boost}")
print(f"CatBoost Accuracy: {accuracy_cat_boost}")


# In[23]:


# Collect accuracies in a dictionary
accuracies = {
    'Logistic Regression': accuracy_log_reg,
    'Random Forest': accuracy_random_forest,
    'Gradient Boosting': accuracy_grad_boost,
    'CatBoost': accuracy_cat_boost
}


# In[25]:


# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red', 'orange'])
plt.ylabel('Accuracy')
plt.title('Classifier Accuracies')
plt.ylim(0.7, 1.0)  # Set y-axis limits for better comparison
for index, value in enumerate(accuracies.values()):
    plt.text(index, value + 0.01, f"{value:.4f}", ha='center')
plt.show()


# In[ ]:




