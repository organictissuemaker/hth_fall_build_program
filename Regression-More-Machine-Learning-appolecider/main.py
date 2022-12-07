import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
%matplotlib inline

''' 
The best way to complete this activity is to copy the code in this file into a jupyter notebook file you create. You will also need to download the csv files with the data. This will allow you to see your outputs easier. Once you have finished with your code, copy it back into this file and submit it to Replit. Don't forget to also upload your new code to Github! 
'''

# Linear Regression Practice

# STEP 1: Import the housing data 
housing_data = 

# STEP 2: Regression is used to determine the relationship between two or more variables within a dataset. We will be using the average area income, the average area house age, the average number of rooms, the average number of bedrooms and the area population to make a prediction about the pricing of houses within an area. Create two variables, x and y, to train your linear regression model.


# STEP 3: Use the appropriate scikit-learn method split your data into the segments that will be used for training and testing. Generally, 75% of your data is used for training and 25% is used to test.
X_train, X_test, y_train, y_test = 


# STEP 4: Create a linear regression model using the data you have manipulated and scikit-learn's LinearRegression feature.

# STEP 5: Can you show the correlation coefficients of all the features in your model?

# STEP 6: Create a visualization (scatterplot, histogram, etc) that shows your models predictions and the actual values.
y_pred = 

# STEP 7: To evaluate the performance of a model more than just the accuracy has to be considered. There are other metrics like the mean absolute error, the mean squared error and the r-squared error. Calculate these three metrics using scikit-learn's metrics import



# Logistic Regression Practice

# STEP 1: Import the titanic data
titanic_data = pd.read_csv('titanic.csv')

# STEP 2: We will be using various features of the titanic data to predict whether or not a passenger survived. Create x and y variables to train your logistic regression model. 

# STEP 3: Use the appropriate scikit-learn method split your data into the segments that will be used for training and testing. Generally, 75% of your data is used for training and 25% is used to test.

# STEP 4: Create a logistic regression model using the data you have manipulated and scikit-learn's LogisticRegression feature.

# STEP 5: Can you show the correlation coefficients of all the features in your model?

# STEP 6: Logistic regression models are evaluated differently than linear regression models. Use the appropriate metrics features from scikit learn to create a report and a matrix for your model