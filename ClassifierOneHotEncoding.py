import re
import pandas as pd
from sklearn import model_selection, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np

fastarecords = []

with open('processed_data.csv', mode='r') as csv_file:
    csv_rows = csv.reader(csv_file)
    for row in csv_rows:
        #print(row)
        fastarecords.append(row)

#remove header
fastarecords.pop(0)

# function to convert a AA sequence string to a numpy array
# converts to lower case, changes any non 'arndcfqeghilkmpstwyv' characters to 'n'
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^arndcfqeghilkmpstwyv]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with AA alphabet
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','r','n','d','c','f','q','e','g','h','i','l','k','m','p','s','t','w','y','v','z']))


def one_hot_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    onehot_encoder = OneHotEncoder(sparse=False, dtype=int, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded = np.delete(onehot_encoded, -1, 1)
    return onehot_encoded


train_df = pd.DataFrame(fastarecords)
del train_df[0]

train_df = train_df[train_df[1].str.len() == 70 ]

print(train_df.head())

train_set = pd.DataFrame()
train_set[1] = train_df.apply(lambda x: one_hot_encoder(string_to_array(x[1])), axis=1)
train_set[2] = train_df[2]

print(train_set.head())

data = train_set.iloc[:, 0].values
print(data)

labels = train_set.iloc[:, 1].values
print(labels)

print(data.shape)
print(labels.shape)

# Splitting the human dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size = 0.20,
                                                    random_state=42)

print(X_train)
print(y_train)


svc = svm.SVC(kernel='linear')

svc.fit(X_train, y_train)

predicted = svc.predict(X_test)
score = svc.score(X_test, y_test)

print('============================================')
print('\nScore ', score)
print('\nResult Overview\n',   metrics.classification_report(y_test, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(y_test, predicted)      )
