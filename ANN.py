#Using tensorflow keras
import tensorflow as tf
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow.keras import layers, models
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

#KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1)

#Encoding Data
ohe = OneHotEncoder()
X_train_enc = ohe.fit_transform(X_train).toarray()

#Lists to store predictions and true values for MSE later
all_predictions = []
all_true_values = []

# Hyperparameter grid for GS
param_grid = {
    'hidden_neurons': [32, 64, 128],  
    'learning_rate': [0.001, 0.01, 0.1]  
}

#Best MSE and model variable to save the best model and use it on the test data
best_mse = float('inf')
best_model = None

#Do KF Cross Validation on training
for train_index, val_index in kf.split(X_train_enc, y_train):
    X_train_fold, X_val_fold = X_train_enc[train_index], X_train_enc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    r_train_fold, r_val_fold = r_train.iloc[train_index], r_train.iloc[val_index]
    
    best_ann_mse = float('inf')
    best_ann = None

    for hidden_neurons in param_grid['hidden_neurons']:
        for learning_rate in param_grid['learning_rate']:
            # Define model
            model = models.Sequential()
            model.add(layers.Dense(hidden_neurons, activation='relu', input_shape=(X_train_fold.shape[1],)))
            model.add(layers.Dense(1, activation='linear'))

            # Compile
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error')

            # Train the model
            model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=32, validation_data=(X_val_fold, y_val_fold))

            # Predict on validation set
            model_val_pred = model.predict(X_val_fold)

            # Evaluate on validation set (MSE for now)
            mse = mean_squared_error(y_val_fold, model_val_pred)
            print(f"Hidden Neurons: {hidden_neurons}, Learning Rate: {learning_rate}, MSE: {mse}")

            # Save the best model
            if mse < best_ann_mse:
                best_ann_mse = mse
                best_ann = model

    # Predict on validation set using the best model
    best_model_val_pred = best_ann.predict(X_val_fold)

    mse = mean_squared_error(y_val_fold, best_model_val_pred)
    print(f"Mean Squared Error on Validation Set: {mse}")

    all_predictions.extend(best_model_val_pred)
    all_true_values.extend(y_val_fold)

    if mse < best_mse:
        best_mse = mse
        best_model = best_ann

#Evaluate on validation set
mse = mean_squared_error(all_true_values, all_predictions)
print(f"Mean Squared Error on ALL Validation Set: {mse}")

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
#Residual plot
model_test_pred_flat = model_test_pred.flatten()
residuals = y_test - model_test_pred_flat
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
#[1]J. Brownlee, “Evaluate the Performance Of Deep Learning Models in Keras,” Machine Learning Mastery, May 25, 2016. https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
#[2]TensorFlow Core, “Introduction to the Keras Tuner | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/keras/keras_tuner
#[3]Tensorflow Core, “Basic regression: Predict fuel efficiency | TensorFlow Core,” TensorFlow. https://www.tensorflow.org/tutorials/keras/regression