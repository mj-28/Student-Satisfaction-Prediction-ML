from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

#AGE MODEL
age_data = pd.read_csv('./data/Age.csv')
age_X = age_data.iloc[:, 1:-1]
age_y = age_data.iloc[:, 0]
age_r = age_data.iloc[:, -1]

age_ohe = OneHotEncoder()
age_X_enc = age_ohe.fit_transform(age_X).toarray()


age_class_weights = compute_sample_weight(class_weight='balanced', y=age_r)

age_best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

age_rf = RandomForestRegressor(**age_best_params) 

age_rf.fit(age_X_enc, age_y, sample_weight=age_class_weights)

#DOMICILE MODEL
dom_data = pd.read_csv('./data/Domicile.csv')
dom_X = dom_data.iloc[:, 1:-1]
dom_y = dom_data.iloc[:, 0]
dom_r = dom_data.iloc[:, -1]

dom_ohe = OneHotEncoder()
dom_X_enc = dom_ohe.fit_transform(dom_X).toarray()

dom_class_weights = compute_sample_weight(class_weight='balanced', y=dom_r)

dom_best_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

dom_rf = RandomForestRegressor(**dom_best_params) 

dom_rf.fit(dom_X_enc, dom_y, sample_weight=dom_class_weights)

#ETHNICITY MODEL
ethn_data = pd.read_csv('./data/Ethnicity.csv')
ethn_X = ethn_data.iloc[:, 1:-1]
ethn_y = ethn_data.iloc[:, 0]
ethn_r = ethn_data.iloc[:, -1]

ethn_ohe = OneHotEncoder()
ethn_X_enc = ethn_ohe.fit_transform(ethn_X).toarray()

ethn_class_weights = compute_sample_weight(class_weight='balanced', y=ethn_r)

ethn_best_params = {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

ethn_rf = RandomForestRegressor(**ethn_best_params) 

ethn_rf.fit(ethn_X_enc, ethn_y, sample_weight=ethn_class_weights)

#DISABILITY STATUS MODEL
ds_data = pd.read_csv('./data/Disability Status.csv')
ds_X = ds_data.iloc[:, 1:-1]
ds_y = ds_data.iloc[:, 0]
ds_r = ds_data.iloc[:, -1]

ds_ohe = OneHotEncoder()
ds_X_enc = ds_ohe.fit_transform(ds_X).toarray()

ds_class_weights = compute_sample_weight(class_weight='balanced', y=ds_r)

ds_best_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}

ds_rf = RandomForestRegressor(**ds_best_params) 

ds_rf.fit(ds_X_enc, ds_y, sample_weight=ds_class_weights)

#SEX MODEL
sex_data = pd.read_csv('./data/Sex.csv')
sex_X = sex_data.iloc[:, 1:-1]
sex_y = sex_data.iloc[:, 0]
sex_r = sex_data.iloc[:, -1]

sex_ohe = OneHotEncoder()
sex_X_enc = sex_ohe.fit_transform(sex_X).toarray()

sex_class_weights = compute_sample_weight(class_weight='balanced', y=sex_r)

sex_best_params = {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

sex_rf = RandomForestRegressor(**sex_best_params) 

sex_rf.fit(sex_X_enc, sex_y, sample_weight=sex_class_weights)

all_user_satisfaction = []
themes = ['Theme 1: Teaching on my course',
            'Theme 2: Learning opportunities',
            'Theme 3: Assessment and feedback',
            'Theme 4: Academic support',
            'Theme 5: Organisation and management',
            'Theme 6: Learning resources',
            'Theme 7: Student voice']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data sent from the client-side JavaScript
    data = request.json
    
    print("Received JSON data:", data)
    # Parse the received data
    characteristic = data['characteristic']
    split = data['split']
    pc = data['pc']
    mos = data['mos']
    los = data['los']
    sub = data['sub']

    uData = pd.DataFrame({
    'Split': [split],
    'Provider country': [pc],
    'Mode of study': [mos],
    'Level of study': [los],
    'Subject': [sub]
    })
    
    # Prepare and encode the user data for each theme
    for theme in themes:
        # Prepare user data with one theme added
        uData_theme = uData.copy()  # Create a copy of the original user data
        uData_theme['Question Number'] = theme  # Add the current theme to the user data
        

        if characteristic == 'Age':
            uData_enc = age_ohe.transform(uData_theme).toarray()
            model = age_rf
        elif characteristic == 'Sex':
            uData_enc = sex_ohe.transform(uData_theme).toarray()
            model = sex_rf
        elif characteristic == 'Ethnicity':
            uData_enc = ethn_ohe.transform(uData_theme).toarray()
            model = ethn_rf
        elif characteristic == 'Domicile':
            uData_enc = dom_ohe.transform(uData_theme).toarray()
            model = dom_rf
        elif characteristic == 'Disability Status':
            uData_enc = ds_ohe.transform(uData_theme).toarray()
            model = ds_rf
        
        
        # Predict the satisfaction for the user data
        user_satisfaction = model.predict(uData_enc)
        
        # Append the predicted satisfaction to the list
        all_user_satisfaction.append(user_satisfaction)

    predictions = []
    for theme, satisfaction in zip(themes, all_user_satisfaction):
        satisfaction_percent = f"{satisfaction[0]:.1f}%"
        predictions.append({'theme': theme, 'satisfaction': satisfaction_percent})
    
    # Return the predicted satisfactions as a JSON response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

#REFERENCES 
#[1]“Tutorial — Flask Documentation (3.0.x),” flask.palletsprojects.com. https://flask.palletsprojects.com/en/3.0.x/tutorial/