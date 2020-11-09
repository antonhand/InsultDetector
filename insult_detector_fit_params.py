__author__ = 'Anton Khandzhyan'

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier

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
        word_vectorizer =   TfidfVectorizer(max_df=0.8, min_df=0.0, strip_accents='unicode', analyzer=stemTokenizer)

        char_vectorizer =  TfidfVectorizer(max_df=0.5, min_df=0.0, analyzer='char_wb', strip_accents='unicode')

        estimators = [
                      ('tfidf1', word_vectorizer),
                      ('tfidf2', char_vectorizer)
                      ]
        combined = FeatureUnion(estimators)
        self.classifier = Pipeline([('vect', combined),
                                    ('clf', SGDClassifier(class_weight='balanced', loss='log', alpha=0.000001, n_jobs=8)), ])

        parameters = {'vect__tfidf1__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
                      'vect__tfidf2__ngram_range': [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)],
                      }
        
        self.fitter = GridSearchCV(self.classifier, parameters, scoring='f1', n_jobs = -1)

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

    def __split_corp_label(self, labeled_discussions):
        corpus = []
        labels = []
        for obj in self.__flatten(labeled_discussions):
            if "insult" in obj:
                corpus.append(obj["text"])
                labels.append(obj["insult"])
        
        return corpus, labels

    
    def fit_params(self, dataset):
        corpus, labels = self.__split_corp_label(dataset)  
        
        self.fitter.fit(corpus, labels)

        print(self.fitter.cv_results_)
        print(self.fitter.best_params_)

        return self.fitter.best_params_


    



