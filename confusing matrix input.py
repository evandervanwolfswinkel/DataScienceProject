import csv
import numpy as np
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

fastarecords = []
#Open the file with the parsed data
with open('processed_data.csv', mode='r') as csv_file:
    csv_rows = csv.reader(csv_file)
    for row in csv_rows:
        fastarecords.append(row)

#remove header
fastarecords.pop(0)

# function to convert a Amino acid sequence string to a numpy array
# converts to lower case, changes any non amino acid characters to 'n'
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^arndcfqeghilkmpstwyv]', 'z', my_string)
    my_aminoarray = np.array(list(my_string))
    return my_aminoarray

# create a label encoder with amino acid alphabet
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','r','n','d','c','f','q','e','g','h','i','l','k','m','p','s','t','w','y','v','z']))

# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

#delete first column from the dataframe as it should not be taken into consideration
dataset = pd.DataFrame(fastarecords)
del dataset[0]

dataset['words'] = dataset.apply(lambda x: getKmers(x[1]), axis=1)
train_set = dataset.drop([1], axis=1)

train_texts = list(train_set['words'])
for item in range(len(train_texts)):
    train_texts[item] = ' '.join(train_texts[item])
y_h = train_set.iloc[:, 0].values

# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
cv = CountVectorizer(ngram_range=(4,4))
train_fit= cv.fit_transform(train_texts)

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(train_fit,
                                                    y_h,
                                                    test_size = 0.20,
                                                    random_state=42)

### Multinomial Naive Bayes Classifier ###
# The alpha parameter was determined by grid search previously
classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Confusion matrix\n")
print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1
accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
