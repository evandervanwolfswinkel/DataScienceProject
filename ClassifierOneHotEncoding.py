import re
import pandas as pd
from sklearn import model_selection, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np
import pickle



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

X = [['a',0],['r',3],['n',1],['d',4],['c',2],['f',0],['q',1],['e',4],['g',2],['h',3],['i',0],['l',0],['k',3],['m',0],['p',2],['s',1],['t',1],['w',0],['y',0],['v',0],['z',10],['u',2]]

train_df = pd.DataFrame(fastarecords)
del train_df[0]

train_df = train_df[train_df[1].str.len() == 70 ]

print(train_df.head())

sequencelist = []
for sequence in train_df[1]:
    sequencelist.append(sequence)

sequencelistarray = []
for sequence in sequencelist:
    sequencelistarray.append(string_to_array(sequence))
#print(sequencelistarray)

onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot_encoder = onehot_encoder.fit(X)
print("catagories")
print(onehot_encoder.categories_)
onehot_encoded = onehot_encoder.fit_transform(sequencelistarray)

labels = train_df.iloc[:, 1].values
print(labels)


svc = svm.SVC(kernel='linear')

svc.fit(onehot_encoded, labels)

model = pickle.dumps(svc)


fastarecords = []

with open('processed_data2.csv', mode='r') as csv_file:
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

X = [['a',0],['r',3],['n',1],['d',4],['c',2],['f',0],['q',1],['e',4],['g',2],['h',3],['i',0],['l',0],['k',3],['m',0],['p',2],['s',1],['t',1],['w',0],['y',0],['v',0],['z',10],['u',2]]

train_df = pd.DataFrame(fastarecords)
del train_df[0]

train_df = train_df[train_df[1].str.len() == 70 ]

print(train_df.head())

sequencelist = []
for sequence in train_df[1]:
    sequencelist.append(sequence)

sequencelistarray = []
for sequence in sequencelist:
    sequencelistarray.append(string_to_array(sequence))
#print(sequencelistarray)

onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot_encoder = onehot_encoder.fit(X)
print("catagories")
print(onehot_encoder.categories_)
onehot_encoded = onehot_encoder.fit_transform(sequencelistarray)

labels = train_df.iloc[:, 1].values
print(labels)


svc = pickle.loads(model)
X_test = onehot_encoded
y_test = labels

predicted = svc.predict(X_test)
score = svc.score(X_test, y_test)

print('============================================')
print('\nScore ', score)
print('\nResult Overview\n',   metrics.classification_report(y_test, predicted))
print('\nConfusion matrix:\n', metrics.confusion_matrix(y_test, predicted)      )
