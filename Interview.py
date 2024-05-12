#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv(r"C:\Users\Zeynep\Downloads\dataset.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Visualize the distribution of features
plt.figure(figsize=(10, 6))
sns.pairplot(data, hue='isVirus')
plt.show()


# In[6]:


# Handling missing values
data.replace(to_replace='659859', value=np.nan, inplace=True)
data.fillna(data.mean(), inplace=True)

# Balancing the classes
from sklearn.utils import resample

# Separate majority and minority classes
majority_class = data[data['isVirus'] == False]
minority_class = data[data['isVirus'] == True]

# Upsample minority class
minority_upsampled = resample(minority_class,
                               replace=True,     # sample with replacement
                               n_samples=len(majority_class),    # to match majority class
                               random_state=42) # reproducible results

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([majority_class, minority_upsampled]).reset_index(drop=True)

# Display new class counts
print("\nBalanced class counts:")
print(data_upsampled['isVirus'].value_counts())

# Visualize the balanced data
plt.figure(figsize=(10, 6))
sns.pairplot(data_upsampled, hue='isVirus')
plt.show()


# In[7]:


# Splitting the data into features and labels
X = data_upsampled.drop('isVirus', axis=1)
y = data_upsampled['isVirus']

# Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a simple machine learning model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Create the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




