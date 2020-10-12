__author__ = 'Anton Khandzhyan'

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
import json

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
        word_vectorizer =   TfidfVectorizer( ngram_range=(1, 3), max_df=0.8, min_df=0.0, strip_accents='unicode', analyzer=stemTokenizer)

        char_vectorizer =  TfidfVectorizer( analyzer='char_wb', max_df=0.5, min_df=0.0, ngram_range=(2, 4), strip_accents='unicode')

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

    def cross_val(self, labeled_discussions):
        corpus = []
        labels = []
        for obj in self.__flatten(labeled_discussions):
            if "insult" in obj:
                corpus.append(obj["text"])
                labels.append(obj["insult"])  
        f1 = cross_val_score(self.classifier, corpus, labels, cv = 3, scoring='f1')
        print("F1-мера: ", f1.mean())
        return f1

if __name__ == "__main__":
    train_data = json.load(open("discussions_tpc_2015/modis/discussions.json", encoding="utf8"))
    test_data = json.load(open("discussions_tpc_2015/students/discussions.json", encoding="utf8"))

    InD = InsultDetector()
    print(InD.cross_val(test_data))

    #InD.train(train_data)
    
    #InD.classify(test_data[:1])
    #print(test_data[0])

    



