import nltk
import random
import re
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

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


#training classifier using Naive-Bayes
classifier=nltk.NaiveBayesClassifier.train(training_set) #training the model
print("Naive Bayes Classifier accuracy percentage: ",(nltk.classify.accuracy(classifier,testing_set))*100) #testing the model




#using our trained classifier to predict whether the majority of tweets collected are positive or negative



#preparation of testing set
csvread = pd.read_csv('astarisborndataset.csv')
d1 = list(csvread['tweet'])
aftm = []   #aftm = tweet - stopwords - punctuations in lower case
atlk=[]     #atlk = after removing emojies and replacing them with ' '
count=0

'''Tokenizing the tweets and converting it to the lower case and removing stopwords and punctuations'''


#nltk.download('punkt')

for tweets in d1:
    wrds = word_tokenize(tweets)
    for w in wrds:
        lw = w.lower()
        if not lw in stopset:
            if not lw in punc:   
                aftm.append(lw) 



''' Removing all kinds of links and emoticons present in the tweet '''
RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)


for tweet in aftm:
    text = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',"",tweet)
    text = '\n'.join([a for a in text.split("\n") if a.strip()])
    rememo = strip_emoji(text)
    atlk.append(rememo)


''' Removing certain kinds of hashtag mentions and symbols present in the tweets'''
hashmensym = ['astarisbornmovie','AStarIsBorn','AStarIsBornMovie','starisbornmovie','astarisborn','//','...','',',']
afht = []       #afht = after removing hashtags
for tweets in atlk:
    if not tweets in hashmensym:
        afht.append(tweets)



testing_set_features={} #extracting testing set features from the twitter data

for everyword in afht:
    testing_set_features[everyword] = everyword in all_words


#classifying using the model that was trained using the Naive Bayes Classifier
sent = classifier.classify(testing_set_features)

print('\n\nThe sentiment of the movie reviews as predicted by our model is : ')
if sent == 'pos':
    print('Positive')
elif sent =='neg':
    print('Negative')
else:
    print('Neutral')
