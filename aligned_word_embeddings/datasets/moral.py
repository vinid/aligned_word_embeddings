import pandas as pd
import numpy as np
from sklearn import preprocessing
import warnings

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

        missing = 0

        virtue_index = (self.mapping_moral_dict[moral_aspect + ".virtue"])
        vice_index = (self.mapping_moral_dict[moral_aspect + ".vice"])

        for index, row in self.data.iterrows():
            if row["aspect"] in [virtue_index, vice_index]:
                try:
                    word_vector = model.wv[row["word"]]
                except:
                    missing = missing + 1
                    continue
                if row["aspect"] == virtue_index:
                    X.append(word_vector)
                    y.append(np.array([1]))
                elif row["aspect"] == vice_index:
                    X.append(word_vector)
                    y.append(np.array([0]))

        warnings.warn("Warning, missing" + str(missing) + " words")
        if normalize:
            X = preprocessing.normalize(np.array(X), norm='l2')

        return shffle(X, y)

class MoralStrength:
    def __init__(self):

        self.lexicon_auto = pd.read_csv(
            "https://raw.githubusercontent.com/oaraque/moral-foundations/master/moralstrength/authority.tsv", sep="\t")
        self.lexicon_fair = pd.read_csv(
            "https://raw.githubusercontent.com/oaraque/moral-foundations/master/moralstrength/fairness.tsv", sep="\t")
        self.lexicon_loy = pd.read_csv(
            "https://raw.githubusercontent.com/oaraque/moral-foundations/master/moralstrength/loyalty.tsv", sep="\t")
        self.lexicon_care = pd.read_csv(
            "https://raw.githubusercontent.com/oaraque/moral-foundations/master/moralstrength/care.tsv", sep="\t")
        self.lexicon_purity = pd.read_csv(
            "https://raw.githubusercontent.com/oaraque/moral-foundations/master/moralstrength/purity.tsv", sep="\t")

    def get_moral(self, model, moral_aspect, normalize = True):
        X = []
        y = []
        missing = 0

        if moral_aspect == "autority":
            lexicon = self.lexicon_auto
        elif moral_aspect == "fairness":
            lexicon = self.lexicon_fair
        elif moral_aspect == "care":
            lexicon = self.lexicon_care
        elif moral_aspect == "purity":
            lexicon = self.lexicon_purity
        elif moral_aspect == "loyality":
            lexicon = self.lexicon_purity
        else:
            raise Exception("Moral aspect not found")

        for index, row in lexicon.iterrows():
            try:
                word_vector = model.wv[row["LEMMA"]]
            except:
                missing = missing + 1
                continue

            X.append(word_vector)
            val = row["EXPRESSED_MORAL"]
            val = (val - 1) / (9 - 1)
            y.append(np.array(val))
        warnings.warn("Warning, missing" + str(missing) + " words")

        X = np.array(X)
        if normalize:
            X = preprocessing.normalize(np.array(X), norm='l2')

        y = np.array(y)
        return shffle(X, y)
