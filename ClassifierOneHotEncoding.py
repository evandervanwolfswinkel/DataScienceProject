import re
import pandas as pd
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


array = string_to_array(fastarecords[1][1])

print(one_hot_encoder(array))

train_df = pd.DataFrame(fastarecords)
del train_df[0]

print(train_df)

train_df['onehotseq'] = train_df.apply(lambda x: one_hot_encoder(string_to_array(x[1])), axis=1)

train_df['onehotlabel'] = train_df.apply(lambda x: one_hot_encoder(string_to_array(x[2])), axis=1)


print(train_df)