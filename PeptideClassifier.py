import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, metrics, model_selection,datasets,linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

####################################
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


# function to convert sequence strings into k-mer words, default size = 6 (hexamer words)
def getKmers(sequence, size=2):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

train_df = pd.DataFrame(fastarecords)
del train_df[0]


train_df['words'] = train_df.apply(lambda x: getKmers(x[1]), axis=1)
train_set = train_df.drop([1], axis=1)
#test_df = pd.DataFrame(X_test)
#test_df['words'] = test_df.apply(lambda x: getKmers(x[1]), axis=1)
#test_set = test_df.drop([1], axis=1)

train_texts = list(train_set['words'])
for item in range(len(train_texts)):
    train_texts[item] = ' '.join(train_texts[item])
y_h = train_set.iloc[:, 0].values         #y_h for test

print(y_h)
#train_texts = list(train_set['words'])
#for item in range(len(train_texts)):
   # train_texts[item] = ' '.join(train_texts[item])
#y_c = train_set.iloc[:, 0].values                       # y_c for train

# Creating the Bag of Words model using CountVectorizer()
# This is equivalent to k-mer counting
# The n-gram size of 4 was previously determined by testing
cv = CountVectorizer(ngram_range=(4,4))
train_fit= cv.fit_transform(train_texts)
#train_fit = cv.transform(train_texts)

# Splitting the human dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(train_fit,
                                                    y_h,
                                                    test_size = 0.20,
                                                    random_state=42)

print(y_train)
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

