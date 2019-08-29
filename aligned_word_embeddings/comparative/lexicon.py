import numpy as np
import pandas as pd
import ujson as json

class Lexicon:
    """
    Wrapper class for lexicon dictionary, allowing threshold selection and split of polarities
    Exposes dict-like methods
    - keys() and values() interfaces
    - key subsetting
    - inclusion operator
    """
    def __init__(self,threshold=0,d=None):
        """
        Constructor
        :param threshold:
        :param d:
        """
        self.threshold = threshold
        
        if d is not None and d is dict:
            self.lex_dict = d
            
            self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
            self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
    
    def load_dict(self,lex_dict):
        """
        Load dictionary as lexicon
        :param lex_dict:
        """
        self.lex_dict = lex_dict
        
        self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
        self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
        
    def load_json(self, fname):
        """
        Load JSON file containing lexicon dictionary
        :param fname:
        """
        with open(fname) as fin:
            self.lex_dict = json.load(fin)
        
        self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
        self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
        
    def load_csv(self, fname, sep=","):
        # WIP
        pass
        df = pd.read_csv(fname, sep, header=0)
        head = list(df.columns)
        if not head == ['word','score']:
            raise RuntimeError("Malformed CSV, expecting 'word' and 'score' columns, got %s instead" % head)
    
    ##########################################
    
    def keys(self):
        return self.lex_dict.keys()
    
    def values(self):
        return self.lex_dict.values()
    
    def __getitem__(self,key):
        return self.lex_dict[key]
    
    def __contains__(self,item):
        return item in self.lex_dict