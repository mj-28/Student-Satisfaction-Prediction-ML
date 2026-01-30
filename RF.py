import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns


#Retrieving Data
data = pd.read_csv('./data/Disability Status.csv')

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
rf = RandomForestRegressor()

# Parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],     # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],    # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],    # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]       # Minimum number of samples required to be at a leaf node
}

#Best MSE and model variable to save the best model and use it on the test data
best_mse = float('inf')
best_model = None
best_param = None

#Do KF Cross Validation on training
for train_index, val_index in kf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]

    #Encoding the X data
    X_train_enc = ohe.fit_transform(X_train_fold).toarray()
    X_val_enc = ohe.transform(X_val_fold).toarray()

    #Class weight based on response
    classW = compute_sample_weight(class_weight='balanced', y=r_train.iloc[train_index])
    
    # Hyperparameter tuning using GridSearchCV 
    # Grid search cant be done outside the loop otherwise we would be contaminating it essentially with the validation data
    grid_search = GridSearchCV(rf, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train_enc, y_train.iloc[train_index], sample_weight=classW)

    # Getting best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_rf = grid_search.best_estimator_

    #Train model
    best_rf.fit(X_train_enc, y_train.iloc[train_index], sample_weight=classW)

    #Predict on validation set
    rf_val_pred = best_rf.predict(X_val_enc)

    #Evaluate on validation set (MSE for now to edit/add all my eval methods)
    mse = mean_squared_error(y_train.iloc[val_index], rf_val_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_model = best_rf
        best_param = best_params


#Test
print("best param ", best_param)
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
#[1]scikit-learn, “3.2.4.3.2. sklearn.ensemble.RandomForestRegressor — scikit-learn 0.20.3 documentation,” Scikit-learn.org, 2018. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
#[2]Geeks For Geeks, “Random Forest Hyperparameter Tuning in Python,” GeeksforGeeks, Dec. 28, 2022. https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/
