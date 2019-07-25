import pandas as pd
import numpy as np
from sklearn import preprocessing

import random


def shffle(X, y):
    data = list(zip(X, y))
    random.shuffle(data)
    X, y = list(zip(*data))
    X = np.array(X)
    y = np.array(y)
    return X, y

class MFDataset:
    def __init__(self, path):
        self.mapping_moral_dict = {
                 'care.virtue': 1,
                 'care.vice': 2,
                 'fairness.virtue': 3,
                 'fairness.vice': 4,
                 'loyalty.virtue': 5,
                 'loyalty.vice': 6,
                 'authority.virtue': 7,
                 'authority.vice': 8,
                 'sanctity.virtue': 9,
                 'sanctity.vice': 10}

        self.data = pd.read_csv(path, sep="\t", names=["word", "aspect"])

    def get_data(self):
        return self.data

    def get_moral(self, model, moral_aspect, normalize = True):
        X = []
        y = []

        virtue_index = (self.mapping_moral_dict[moral_aspect + ".virtue"])
        vice_index = (self.mapping_moral_dict[moral_aspect + ".vice"])

        for index, row in self.data.iterrows():
            if row["aspect"] in [virtue_index, vice_index]:
                try:
                    word_vector = model.wv[row["word"]]
                except:
                    continue
                if row["aspect"] == virtue_index:
                    X.append(word_vector)
                    y.append(np.array([1]))
                elif row["aspect"] == vice_index:
                    X.append(word_vector)
                    y.append(np.array([0]))
        if normalize:
            X = preprocessing.normalize(np.array(X), norm='l2')

        return shffle(X, y)
