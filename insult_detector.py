__author__ = 'Anton Khandzhyan'

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
import json


class InsultDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        estimators = [
                      ('tfidf1', TfidfVectorizer( ngram_range=(1, 3), max_df=0.8, min_df=0.0, strip_accents='unicode' )),
                      ('tfidf2', TfidfVectorizer( analyzer='char_wb', max_df=0.5, min_df=0.0, ngram_range=(2, 6), strip_accents='unicode' ))
                      ]
        combined = FeatureUnion(estimators)
        self.classifier = Pipeline([('vect', combined),
                                        ('clf', SGDClassifier(n_jobs = 8)), ])

    

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

    InD = InsultDetector()
    print(InD.cross_val(train_data))
    



