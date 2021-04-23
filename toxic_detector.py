__author__ = 'Anton Khandzhyan'

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import json
import pandas as pd

import nltk

stem = nltk.stem.snowball.RussianStemmer()

def stemTokenizer(text):
    tokens = []
    words = nltk.tokenize.WordPunctTokenizer().tokenize(text)
    for word in words:
        tokens.append(stem.stem(word))
    return tokens


class ToxicDetector:

    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """ 
        word_vectorizer =   TfidfVectorizer( ngram_range=(1, 3), max_df=0.8, min_df=0.0, strip_accents='unicode', analyzer=stemTokenizer)

        char_vectorizer =  TfidfVectorizer( analyzer='char_wb', max_df=0.5, min_df=0.0, ngram_range=(1, 8), strip_accents='unicode')

        estimators = [
                      ('tfidf1', word_vectorizer),
                      ('tfidf2', char_vectorizer)
                      ]
        combined = FeatureUnion(estimators)
        self.classifier = Pipeline([('vect', combined),
                                    ('clf', SGDClassifier(class_weight='balanced', loss='log', alpha=0.000001,  n_jobs=-1)), ])

    

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
        # corpus = []
        # labels = []
        # for obj in self.__flatten(labeled_discussions):
        #     if "insult" in obj:
        #         corpus.append(obj["text"])
        #         labels.append(obj["insult"])
        
        corpus = labeled_discussions["comment_text"].values
        labels = labeled_discussions["toxic"].values

        return corpus, labels

    def train(self, labeled_discussions):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """
        corpus, labels = self.__split_corp_label(labeled_discussions)        

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
        corpus, labels = self.__split_corp_label(labeled_discussions)

        val = cross_validate(self.classifier, corpus, labels,  scoring=['f1', 'precision', 'recall' ], n_jobs=-1)
        
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


if __name__ == "__main__":
    train_df = pd.read_csv("rutoxic/labeled.csv", encoding="utf8")
    # test_data = json.load(open("discussions_tpc_2015/modis/discussions.json", encoding="utf8"))
    # train_data = json.load(open("discussions_tpc_2015/students/discussions.json", encoding="utf8"))

    TOXITY_TYPE = "toxic"

    df = pd.read_csv("jigsaw/jigsaw-toxic-comment-train-google-ru.csv", encoding="utf8")

    df = df[(df[TOXITY_TYPE] == "1") | (df[TOXITY_TYPE] == "0") | (df[TOXITY_TYPE] == 1) | (df[TOXITY_TYPE] == 0)]
    df.toxic = df[TOXITY_TYPE].astype(int)

    toxic_df = df[df[TOXITY_TYPE] == 1]
    nottoxic_df = df[df[TOXITY_TYPE] == 0]

    toxic_l = len(toxic_df)
    nottoxic_l = len(nottoxic_df)


    df = pd.concat([toxic_df, nottoxic_df[:toxic_l]])
    test_df = df


    ToxD = ToxicDetector()
    #print(ToxD.cross_val(test_data))
    ToxD.evaluate(train_df, test_df)

    #ToxD.train(train_data)
    
    #ToxD.classify(test_data[:1])
    #print(test_data[0])

    



