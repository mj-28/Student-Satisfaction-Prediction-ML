import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import numpy as np

#Change name accordingly to data file to be explored
data = pd.read_csv('./data/Age.csv')

#Data Description
#Outputs it to a file description.txt => headers arent aligned great but its understandable
outputFile = 'description.txt'

with open(outputFile, 'w') as file:
    file.write(data.describe(include='all').to_string()) 

#Finding unique values
def uniquevalues(data):
    for column in data.columns.tolist():
        print(column + ': ' + str(len(data[column].unique())))
uniquevalues(data)

#Fiding missing values
def findmissingvalues(data):
    data.replace('?', pd.NA, inplace=True)
    print(data.isna().sum())
findmissingvalues(data)

#No feature selection required