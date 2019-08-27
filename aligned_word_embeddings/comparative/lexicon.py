import numpy as np
import pandas as pd
import ujson as json

class Lexicon:
    
    def __init__(self,threshold=0,d=None):        
        self.threshold = threshold
        
        if d is not None and d is dict:
            self.lex_dict = d
            
            self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
            self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
    
    def load_dict(self,lex_dict):
        self.lex_dict = lex_dict
        
        self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
        self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
        
    def load_json(self, fname):
        with open(fname) as fin:
            self.lex_dict = json.load(fin)
        
        self.positive = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] > self.threshold}
        self.negative = {k: self.lex_dict[k] for k in self.lex_dict.keys() if self.lex_dict[k] < self.threshold}
        
    def load_csv(self, fname, sep=","):
        pass
    
    ##########################################
    
    def keys(self):
        return self.lex_dict.keys()
    
    def values(self):
        return self.lex_dict.values()
    
    def __getitem__(self,key):
        return self.lex_dict[key]
    
    def __contains__(self,item):
        return item in self.lex_dict