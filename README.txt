Welcome! 
This is all the code and data utilised for the study "Predicting student satisfaction using machine learning". The majority of the code is written in python with some code and javascript and css for the website. The code was built and ran in VSCode

NSS23_characteristics_workbook2211
=> This excel file is the original NSS data downloaded from [1]

Data Folder
=> The data folder includes the specific datasets used to run the code, the feature have been manually selected and each characteristic has been made into an individual file each

Library/Modules required to run the code
=>In order to run the code a few libraries and modules are required

==>matplotlib
==>pandas
==>scikit learn (sklearn)
==>numpy
==>flask
==>seaborn
==>tensorflow

Data Exploration 
=> The data explorartion python file is the data exploration befor building the models
=> The 'description.txt' file is the file made when this code is run, it gives the information of the data exploration

Machine Learning Models
=>All the models are ran in python. 
=>They have mostly the same skeleton code adapted to that model.
=>You can run them in VSCode python terminal, make sure the terminal's destination is the folder they are in. 

Running the interface
=>To run the interface it must be run locally
=>Open 'app.py' and run it in the python terminal, it may take 10 to 20 minutes
=>Click on the local http to run it in a browser
=>You may need to wait for the webpage to load again
=>The slow running of the interface is due to the fact it has to train multiple ML algorithms
=>Once loaded READ THE WARNING, then you may interact with the webpage however you wish 

Static Folder
=>The static folder includes the javascript and the css for the website
=> Flask in python (app.py) was used to run the backend and javascript the front end

Templates folder
=> Has the index.html file for the website

Model Results:
=> The model result excel file contains all the tables used to analyse/avg the data. 

References
=>References specific to code can be found at the bottom of that coding file. 
=>Some of the references bellow are for the plots, they were used with all ML algorithms so to avoid having them in each file they are put in general here.

[1] Office for Students, “Download the NSS data - Office for Students,” www.officeforstudents.org.uk, Aug. 10, 2023. Available: https://www.officeforstudents.org.uk/data-and-analysis/national-student-survey-data/download-the-nss-data/

[2]Geeks For Geeks, “How to Create a Residual Plot in Python,” GeeksforGeeks, Feb. 17, 2022. https://www.geeksforgeeks.org/how-to-create-a-residual-plot-in-python/

[3]Scikit Learn, “Plotting Cross-Validated Predictions,” scikit-learn. https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#sphx-glr-auto-examples-model-selection-plot-cv-predict-py




