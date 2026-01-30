# Predicting Student Satisfaction Using Machine Learning 

## Overview 
This project applies and evaluates multiple **machine learning models** to predict student satisfaction in higher education using demographic and institutional data from the **UK National Student Survey (NSS) 2023**.

The goal is to assess which modelling approaches perform best for predicting satisfaction outcomes, and to explore how such models could be deployed in an interactive system that allows students to estimate satisfaction based on their own characteristics.

##Data
The study uses data from the **National Student Survey (NSS) 2023** (NSS23_characteristics_workbook2211) [1], focusing on student demographic characteristics including:
- Sex
- Age
- Domicile
- Ethnicity
- Disability status

## Machine Learning Models
The following supervised learning models were implemented and evaluated:

- Random Forest (RF)
- Artificial Neural Network (ANN)
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Linear Regression (LR)
- Naïve Bayes (NB)

All models were evaluated using appropriate regression and classification metrics alongside diagnostic plots to assess model performance and behaviour.

## Results
Among the evaluated models, **Random Forest** achieved the strongest predictive performance on the NSS dataset.

Model performance was analysed using quantitative evaluation metrics and visual diagnostics to assess:
- Predictive accuracy
- Error distributions
- Model stability across demographic groups

Results and detailed analysis are presented in the accompanying dissertation paper.

## Interactive Interface
A lightweight **web-based interface** was developed to demonstrate how trained models could be deployed in practice.

- Backend: Python (Flask)
- Frontend: HTML, CSS, JavaScript
- Users can input demographic characteristics to generate predicted satisfaction outcomes

User feedback suggested positive perceived value in such an interface, highlighting its potential as a decision-support or exploratory tool.

## Ethical Considerations
The project explicitly considers issues of **algorithmic accountability and fairness**, particularly in the context of demographic-based predictions.

Limitations related to sample size, model bias, and interpretability are discussed, alongside recommendations for improving fairness-aware modelling and interface design.

---
## Repository Structure
###Data Folder
=> The data folder includes the specific datasets used to run the code, the feature have been manually selected and each characteristic has been made into an individual file each

###Library/Modules required to run the code
=>In order to run the code a few libraries and modules are required

==>matplotlib
==>pandas
==>scikit learn (sklearn)
==>numpy
==>flask
==>seaborn
==>tensorflow

### Data Exploration 
=> The data explorartion python file is the data exploration before building the models
=> The 'description.txt' file is the file made when this code is run, it gives the information of the data exploration

###Machine Learning Models
=>All the models are ran in python. 
=>They have mostly the same skeleton code adapted to that model.
=>You can run them in VSCode python terminal, make sure the terminal's destination is the folder they are in. 

###Running the interface
=>To run the interface it must be run locally
=>Open 'app.py' and run it in the python terminal, it may take 10 to 20 minutes
=>Click on the local http to run it in a browser
=>You may need to wait for the webpage to load again
=>The slow running of the interface is due to the fact it has to train multiple ML algorithms
=>Once loaded READ THE WARNING, then you may interact with the webpage however you wish 

###Static Folder
=>The static folder includes the javascript and the css for the website
=> Flask in python (app.py) was used to run the backend and javascript the front end

###Templates folder
=> Has the index.html file for the website

###Model Results:
=> The model result excel file contains all the tables used to analyse/avg the data. 

##References
=>References specific to code can be found at the bottom of that coding file. 
=>Some of the references bellow are for the plots, they were used with all ML algorithms so to avoid having them in each file they are put in general here.

[1] Office for Students, “Download the NSS data - Office for Students,” www.officeforstudents.org.uk, Aug. 10, 2023. Available: https://www.officeforstudents.org.uk/data-and-analysis/national-student-survey-data/download-the-nss-data/

[2]Geeks For Geeks, “How to Create a Residual Plot in Python,” GeeksforGeeks, Feb. 17, 2022. https://www.geeksforgeeks.org/how-to-create-a-residual-plot-in-python/

[3]Scikit Learn, “Plotting Cross-Validated Predictions,” scikit-learn. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py




