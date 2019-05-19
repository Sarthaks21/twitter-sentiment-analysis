import nltk
import random
import re
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify.scikitlearn import SklearnClassifier

#nltk.download('movie_reviews')
#nltk.download('stopwords')

documents= [(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

stopset=set(stopwords.words("english")) #set of all predefined stopword
punc=['!','@','#','$','%','^','&','*','(',')','-','_','+','=','{','}',':',';','<','>',',','.','?','/','~','`','|']  #set of all possible punctuations


all_words = []
fil_words=[]


for w in movie_reviews.words():
    all_words.append(w.lower())     #converting all words to lowercase


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys()) [:3000]
word_features=[w for w in word_features if w not in stopset]    #filter all stopwords
word_features=[w for w in word_features if w not in punc]   #filter all punctuations


def find_features(document):
    words=set(document)
    features= {}
    for w in word_features:
        features[w]=w in words
    return features


featuresets=[(find_features(rev),category) for (rev,category) in documents]  
training_set=featuresets[:1900]
testing_set=featuresets[1900:]



#training classifiers

classifier=nltk.NaiveBayesClassifier.train(training_set) #training the model
print("Naive Bayes Classifier accuracy percentage: ",(nltk.classify.accuracy(classifier,testing_set))*100) #testing the model
nb_acc = nltk.classify.accuracy(classifier,testing_set)*100

#classifier1 = normal SVC
classifier1 = SklearnClassifier(SVC())
classifier1.train(training_set)
print('SVC Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier1, testing_set))*100)
svc_acc = nltk.classify.accuracy(classifier1, testing_set)*100

#classifier2 = LinearSVC
classifier2 = SklearnClassifier(LinearSVC())
classifier2.train(training_set)
print('LinearSVC Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier2, testing_set))*100)
lsvc_acc = nltk.classify.accuracy(classifier2, testing_set)*100

#classifier3 = NuSVC
classifier3 = SklearnClassifier(NuSVC())
classifier3.train(training_set)
print('NuSVC Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier3, testing_set))*100)
nusvc_acc = nltk.classify.accuracy(classifier3, testing_set)*100

#classifier4 = DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
classifier4 = SklearnClassifier(DecisionTreeClassifier())
classifier4.train(training_set)
print('Decision Tree Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier4, testing_set))*100)
dt_acc = nltk.classify.accuracy(classifier4, testing_set)*100
    
#classifier5 = LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier5 = SklearnClassifier(LogisticRegression())
classifier5.train(training_set)
print('Logistic Regression Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier5, testing_set))*100)
lr_acc = nltk.classify.accuracy(classifier5, testing_set)*100
    
#classifier6 = SGDClassifier
from sklearn.linear_model import SGDClassifier
classifier6 = SklearnClassifier(SGDClassifier())
classifier6.train(training_set)
print('\n\nSGD Classifier accuracy percentage: ', (nltk.classify.accuracy(classifier6, testing_set))*100)
sgd_acc = nltk.classify.accuracy(classifier6, testing_set)*100




#plotting graph
import matplotlib.pyplot as plt

y_axis = ['Naive Bayes', 'SVC', 'LinearSVC', 'NuSVC', 'DecisionTree', 'LogisticRegression', 'SGD']
x_axis = [nb_acc, svc_acc, lsvc_acc, nusvc_acc, dt_acc, lr_acc, sgd_acc]
fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.bar(y_axis, x_axis, label = "Classifier v/s Accuracy Graph")
plt.show()

