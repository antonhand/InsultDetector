__author__ = 'Anton Khandzhyan'

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics

import nltk

stem = nltk.stem.snowball.RussianStemmer()

def stemTokenizer(text):
    tokens = []
    words = nltk.tokenize.WordPunctTokenizer().tokenize(text)
    for word in words:
        tokens.append(stem.stem(word))
    return tokens


class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """ 
        word_vectorizer =   TfidfVectorizer( ngram_range=(1, 2), max_df=0.8, min_df=0.0, strip_accents='unicode', analyzer=stemTokenizer)

        char_vectorizer =  TfidfVectorizer( analyzer='char_wb', max_df=0.5, min_df=0.0, ngram_range=(1, 5), strip_accents='unicode')

        estimators = [
                      ('tfidf1', word_vectorizer),
                      ('tfidf2', char_vectorizer)
                      ]
        combined = FeatureUnion(estimators)
        self.classifier = Pipeline([('vect', combined),
                                    ('clf', SGDClassifier(class_weight='balanced', loss='log', alpha=0.000001,  n_jobs=8)), ])

    

    def __flatten_child(self, trees):
        flat = []
        flat.extend(trees)
        for obj in trees:
            if "children" in obj:
                flat.extend(self.__flatten_child(obj["children"]))
        return flat

    def __flatten(self, trees):
        flat = []
        for root in trees:
            obj = root["root"]
            if "children" in obj:
                flat.extend(self.__flatten_child(obj["children"]))
        return flat

    def train(self, labeled_discussions):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """
        corpus = []
        labels = []
        for obj in self.__flatten(labeled_discussions):
            if "insult" in obj:
                corpus.append(obj["text"])
                labels.append(obj["insult"])        
                
        self.classifier.fit(corpus, labels)
        print("Классификатор успешно натренирован")

    def classify(self, unlabeled_discussions):
        """
        This method take the list of discussions as input. You should predict for every message in every
        discussion (except root) if the message insult or not. Than you should replace the value of field "insult"
        for True if the method is insult and False otherwise.
        :param discussion: list of discussion. The field insult would be replaced by False.
        :return: None
        """
        for obj in self.__flatten(unlabeled_discussions):

            ins = self.classifier.predict([obj["text"]])[0]
            obj["insult"] = ins

        return unlabeled_discussions
    
    def __split_corp_label(self, labeled_discussions):
        corpus = []
        labels = []
        for obj in self.__flatten(labeled_discussions):
            if "insult" in obj:
                corpus.append(obj["text"])
                labels.append(obj["insult"])
        
        return corpus, labels
            

    def cross_val(self, labeled_discussions):
        corpus, labels = self.__split_corp_label(labeled_discussions)

        val = cross_validate(self.classifier, corpus, labels, cv = 3, scoring=['f1', 'precision', 'recall' ])
        
        print("F1: ", val['test_f1'].mean())
        print("precision: ", val['test_precision'].mean())
        print("recall: ", val['test_recall'].mean())
        
        return val

    def evaluate(self, train_discussions, test_discussions):
        self.train(train_discussions)

        corpus, labels_true = self.__split_corp_label(test_discussions)
        labels_pred = self.classifier.predict(corpus)
        report = metrics.classification_report(labels_true, labels_pred, digits = 4)
        print(report)

        return report



    



