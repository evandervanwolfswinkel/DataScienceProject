import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics, model_selection,datasets,linear_model
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

####################################
fastarecords = []

with open('processed_data.csv', mode='r') as csv_file:
    csv_rows = csv.reader(csv_file)
    for row in csv_rows:
        #print(row)
        fastarecords.append(row)

#remove header
fastarecords.pop(0)

X_train, X_test = model_selection.train_test_split(
    fastarecords, train_size=0.75,test_size=0.25)
print ("X_train: ", X_train)
print("X_test: ", X_test)