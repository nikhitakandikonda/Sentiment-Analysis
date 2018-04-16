import os
import glob
import os.path
import re
import nltk
import sklearn
from os.path import join
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from functools import partial
from nltk.tokenize import SpaceTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import partial
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
# from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# from sklearn.model_selection import cross_val_score
# https://datawarrior.wordpress.com/2015/08/12/codienerd-1-r-or-python-on-text-mining/
os.chdir('C:\\Users\\Manu\\Downloads\\Courses\\Introduction to Data Mining\\Twitter Sentiment Analysis\\aclImdb')

tokenizer = SpaceTokenizer()
lemmatizer = WordNetLemmatizer()
cachedStopWords = stopwords.words("english")

reviews_list = ['test_pos', 'test_neg', 'train_neg', 'train_pos']
for name in reviews_list:
    path = os.getcwd() + '\\' + name.replace('_', '\\')
    exec(name + '=[]')
    exec(name + '_reviews = []')
    exec(name + '_ratings = []')

    for file in glob.glob(os.path.join(path, '*.txt')):
        with open(file, encoding="utf8") as f:
            text = f.read()
        rating_val = re.split("[._]", file)[1]
        exec(name + '_reviews.append(text)')
        exec(name + '_ratings.append(rating_val)')

reviews_train = train_pos_reviews + train_neg_reviews
ratings_train = train_pos_ratings + train_neg_ratings
reviews_train = [tokenizer.tokenize(x) for x in reviews_train]

reviews_test = test_pos_reviews + test_neg_reviews
ratings_test = test_pos_ratings + test_neg_ratings
reviews_test = [tokenizer.tokenize(x) for x in reviews_test]

pipeline = [lambda s: re.sub('[^\w\s]', '', s),
            lambda s: re.sub('[\d]', '', s),
            lambda s: s.lower(),
            lambda s: list(filter(lambda s: not (s in cachedStopWords), tokenizer.tokenize(s))),
            lambda s: list(map(lambda s: lemmatizer.lemmatize(s), s)),
            lambda s: list(filter(None, s))
            ]


def preprocess_text(text, pipeline):
    if len(pipeline) == 0:
        return text
    else:
        return preprocess_text(pipeline[0](text), pipeline[1:])


train_preprocessed_pos = list(map(partial(preprocess_text, pipeline=pipeline), train_pos_reviews))
train_preprocessed_neg = list(map(partial(preprocess_text, pipeline=pipeline), train_neg_reviews))

train_pos_words = [y for x in train_preprocessed_pos for y in x]
train_neg_words = [y for x in train_preprocessed_neg for y in x]

pos_freq = nltk.FreqDist(train_pos_words)
neg_freq = nltk.FreqDist(train_neg_words)

print("Filtering starts here")
pos_freq = {key: val for key, val in pos_freq.items() if val > 100}
print("Filtering starts here")
neg_freq = {key: val for key, val in neg_freq.items() if val > 100}

print("Intersection of words")
words = list(set(pos_freq.keys()) | set(neg_freq.keys()))

print("Probabilities of words")
probs = {}
for word in words:
    if ((word in pos_freq.keys()) & (word in neg_freq.keys())):
        probs[word] = (pos_freq[word] * neg_freq[word]) / (pos_freq[word] + neg_freq[word]) ** 2
    else:
        probs[word] = 0

prob_threshold = 0.24
words_list = [key for key, val in probs.items() if val < prob_threshold]


def find_features(document):
    words = set(document)
    features = {}
    for w in words_list:
        features[w] = w in words
    return features


ratings_train_data = []
ratings_train = list(map(int, ratings_train))
for rating in ratings_train:
    if rating <= 5:
        ratings_train_data.append("Negative")
    else:
        ratings_train_data.append("Positive")

ratings_test_data = []
ratings_test = list(map(int, ratings_test))
for rating in ratings_test:
    if rating <= 4:
        ratings_test_data.append("Negative")
    else:
        ratings_test_data.append("Positive")

reviews_all_train = list(zip(reviews_train, ratings_train_data))
reviews_all_test = list(zip(reviews_test, ratings_test_data))

train = [(find_features(review), rating) for (review, rating) in reviews_all_train]
test = [(find_features(review), rating) for (review, rating) in reviews_all_test]

reviews = train + test
ratings = ratings_train_data + ratings_test_data
print(len(reviews), len(ratings))

reviews_sublists = [reviews[i:i + 5000] for i in range(0, len(reviews), 5000)]
ratings_sublists = [ratings[i:i + 5000] for i in range(0, len(ratings), 5000)]
print(len(reviews_sublists), len(ratings_sublists))

MNB_classifier = SklearnClassifier(MultinomialNB())
BNB_classifier = SklearnClassifier(BernoulliNB())
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SVC_classifier = SklearnClassifier(SVC())
LinearSVC_classifier = SklearnClassifier(LinearSVC())
NuSVC_classifier = SklearnClassifier(NuSVC())
MNB_Scores = []
BNB_Scores = []
LR_Scores = []
SGD_Scores = []
SVC_Scores = []
LSVC_Scores = []
NuSVC_Scores = []
kfolds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for k in range(0, 10):
    train_kfolds = []
    ratings_train_kfolds = []
    for l in range(0, 10):
        if l != k:
            train_kfolds = train_kfolds + reviews_sublists[l]
            ratings_train_kfolds = ratings_train_kfolds + ratings_sublists[l]
        else:
            test_kfolds = reviews_sublists[l]
            ratings_test_kfolds = ratings_sublists[l]
    print("This is the iteration for value ", k, len(train_kfolds), len(ratings_train_kfolds), len(test_kfolds),
          len(ratings_test_kfolds))

    MNB_classifier.train(train)
    MNB_Scores.append(nltk.classify.accuracy(MNB_classifier, test))

    BNB_classifier.train(train)
    BNB_Scores.append(nltk.classify.accuracy(BNB_classifier, test))

    LogisticRegression_classifier.train(train)
    LR_Scores.append(nltk.classify.accuracy(LogisticRegression_classifier, test))

    SGDClassifier_classifier.train(train)
    SGD_Scores.append(nltk.classify.accuracy(SGDClassifier_classifier, test))

    SVC_classifier.train(train)
    SVC_Scores.append(nltk.classify.accuracy(SVC_classifier, test))

    LinearSVC_classifier.train(train)
    LSVC_Scores.append(nltk.classify.accuracy(LinearSVC_classifier, test))

    NuSVC_classifier.train(train)
    NuSVC_Scores.append(nltk.classify.accuracy(NuSVC_classifier, test))

print(sum(MNB_Scores) / 10)
print(sum(BNB_Scores) / 10)
print(sum(LR_Scores) / 10)
print(sum(SGD_Scores) / 10)
print(sum(SVC_Scores) / 10)
print(sum(LSVC_Scores) / 10)
print(sum(NuSVC_Scores) / 10)


