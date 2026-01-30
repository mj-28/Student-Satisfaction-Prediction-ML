#Import modules
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns


#Retrieving Data
data = pd.read_csv('./data/Sex.csv')

X = data.iloc[:, 1:-1]  #All data
y = data.iloc[:, 0]  #Target coloumn
r = data.iloc[:, -1] #Responses that will be used as a class weight

#Split data
#Training and Test
X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X, y, r, test_size=0.2, random_state = 1)#Had to remove stratification for now

#KFold (Not stratified because its a regression task)
kf = KFold(n_splits=5, shuffle=True, random_state=1)

#Encoding Data
ohe = OneHotEncoder()

#Model
knn = KNeighborsRegressor()

#Parameter grid for grid search
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
}

#Best MSE and model variable to save the best model and use it on the test data
best_mse = float('inf')
best_model = None

#Do SKF Cross Validation on training
for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]

    #Encoding the X data
    X_train_enc = ohe.fit_transform(X_train_fold).toarray()
    X_val_enc = ohe.transform(X_val_fold).toarray()

    # Hyperparameter tuning using GridSearchCV for KNN
    grid_search = GridSearchCV(knn, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train_enc, y_train.iloc[train_index])

    # Get the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_knn = grid_search.best_estimator_

    # Predict on validation set using the best model
    knn_val_pred = best_knn.predict(X_val_enc)

    # Train model
    # knn.fit(X_train_enc, y_train.iloc[train_index])

    #Evaluate on validation set
    mse = mean_squared_error(y_train.iloc[val_index], knn_val_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_model = best_knn

#Test
X_test_enc = ohe.transform(X_test).toarray()
model_test_pred = best_model.predict(X_test_enc)

#Evaluate
mse_test = mean_squared_error(y_test, model_test_pred)
mae_test = mean_absolute_error(y_test, model_test_pred)
r2_test = r2_score(y_test, model_test_pred)

print(f"Mean Squared Error on Test Set: {mse_test}")
print(f"Mean Absolute Error on Test Set: {mae_test}")
print(f"R-squared on Test Set: {r2_test}")

#Plots
# Residual plot
residuals = y_test - model_test_pred
plt.figure(figsize=(8, 6))
sns.residplot(x=model_test_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.title('Residual Plot')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

# Prediction error plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, model_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Prediction Error Plot')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()

###REFERENCES###
#[1]Analytics Vidhya, “A Practical Introduction to K-Nearest Neighbor for Regression,” Analytics Vidhya, May 07, 2019. https://www.analyticsvidhya.com/blog/2018/08/k-nearest-neighbor-introduction-regression-python/
#[2]Scikit Learn, “sklearn.neighbors.KNeighborsRegressor — scikit-learn 0.22 documentation,” Scikit-learn.org, 2019. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html