import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns

#Retrieving Data
data = pd.read_csv('./data/Age.csv')

X = data.iloc[:, 1:-1]  #All data
y = data.iloc[:, 0]  #Target coloumn
r = data.iloc[:, -1] #Responses that will be used as a class weight

#Split data
#Training and Test
X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X, y, r, test_size=0.2, random_state = 1)#Had to remove stratification for now

#Encoding Data
ohe = OneHotEncoder()
ss = StandardScaler()

X_train_enc = ohe.fit_transform(X_train)
X_test_enc = ohe.fit_transform(X_test)

#Making y discrete - discretisation process
n_bins = 10
kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
y_train_d = kbins.fit_transform(y_train.values.reshape(-1, 1))
y_test_d = kbins.transform(y_test.values.reshape(-1, 1))

#KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

#Class weight
classW = compute_sample_weight('balanced', r_train)

#Model
gnb = GaussianNB()

#Parameter grid for grid search
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5] 
}

#Best MSE and model variable to save the best model and use it on the test data
best_mse = float('inf')
best_model = None

#Do KF Cross Validation on training
for train_index, val_index in kf.split(X_train_enc, y_train_d):
    X_train_fold, X_val_fold = X_train_enc[train_index], X_train_enc[val_index]
    y_train_fold, y_val_fold = y_train_d[train_index], y_train_d[val_index]
    r_train_fold, r_val_fold = r_train.iloc[train_index], r_train.iloc[val_index]

    # Hyperparameter tuning using GridSearchCV for Gaussian NB
    grid_search = GridSearchCV(gnb, param_grid, scoring='neg_mean_squared_error', cv=3)
    grid_search.fit(X_train_fold.toarray(), y_train_fold.ravel(), sample_weight=classW[train_index])

    # Get the best hyperparameters and the best model
    best_params = grid_search.best_params_
    best_gnb = grid_search.best_estimator_

    # Predict on validation set using the best model
    gnb_val_pred = best_gnb.predict(X_val_fold.toarray())

    #Evaluate on validation set (MSE for now to edit/add all my eval methods)
    mse = mean_squared_error(y_val_fold, gnb_val_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")
    if mse < best_mse:
        best_mse = mse
        best_model = best_gnb


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
#[1]C.-N. Hsu, H.-J. Huang, and T.-T. Wong, “Implications of the Dirichlet Assumption for Discretization of Continuous Variables in Naive Bayesian Classifiers,” Machine Learning, vol. 53, no. 3, pp. 235–263, 2003, doi: https://doi.org/10.1023/a:1026367023636. Available: https://link.springer.com/article/10.1023%2FA%3A1026367023636. 
#[2]Scikit Learn, “sklearn.naive_bayes.GaussianNB — scikit-learn 0.22.1 documentation,” scikit-learn.org. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
