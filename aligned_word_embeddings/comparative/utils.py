import itertools
import numpy as np
from spm1d.stats import hotellings2

_MEASURES = ["jaccard", "count", "raw"]

def vocabulary_overlap(mod_a, mod_b, measure="jaccard"):
    """
    Returns overlap in vocabulary between models mod_a ad mod_b
    :param mod_a:
    :param mod_b:
    :param measure:
    :return:
    """
    if measure not in _MEASURES:
        raise RuntimeError("Unknown overlap measure [use %s]" % ", ".join(_MEASURES))
    
    if measure == "jaccard":
        voc_a = set( mod_a.wv.vocab.keys() )
        voc_b = set( mod_b.wv.vocab.keys() )
        return len(  voc_a.intersection( voc_b ) ) / len( voc_a.union( voc_b ) )
    
    # A inters B
    if measure == "count":
        voc_a = set( mod_a.wv.vocab.keys() )
        voc_b = set( mod_b.wv.vocab.keys() )
        return len(  voc_a.intersection( voc_b ) )
    
    if measure == "raw":
        voc_a = set( mod_a.wv.vocab.keys() )
        voc_b = set( mod_b.wv.vocab.keys() )
        return voc_a.intersection( voc_b )
    
def vocabulary_report(aligned, measure="jaccard", internal=False):
    """
    Prints a report of vocabulary overlaps for all trained slices
    :param aligned:
    :param measure:
    :param internal:
    :return:
    """
    if measure == "raw": RuntimeError("Cannot report on raw vocabularies")
    if measure not in _MEASURES:
        raise RuntimeError("Unknown overlap measure [use %s]" % ", ".join(_MEASURES))
    print("Computing %s vocabulary reports"%measure)
    
    if internal:
        for a,b in itertools.combinations(aligned.trained_slices.items()):
            print("%s - %s: " % (a[0],b[0]), vocabulary_overlap(a[1],b[1],measure=measure) )
        return
    else:
        for n,s in aligned.trained_slices.items():
            print(n+": ", vocabulary_overlap(s,aligned.compass,measure=measure))
        return

def frequency_rate(mod_a, mod_b):
    """
    Returns frequency rates of model mod_a over model mod_b
    :param mod_a:
    :param mod_b:
    :return:
    """
    tot_a = sum([x.count for x in mod_a.wv.vocab.values()])
    tot_b = sum([x.count for x in mod_b.wv.vocab.values()])
    return tot_a / tot_b

def frequency_report(aligned, internal=False):
    """
    Prints report of frequency rates for all trained slices
    :param aligned:
    :param internal:
    :return:
    """
    if internal:
        for a,b in itertools.combinations(aligned.trained_slices.items()):
            print("%s / %s: " % (a[0],b[0]), frequency_rate(a[1],b[1]) )
        return
    else:
        for n,s in aligned.trained_slices.items():
            print("%s / compass: " % n,  frequency_rate(s,aligned.compass) )
        return


def parse_lexicon(model, lexicon, sanitize=True):
    """
    Returns tuple composed of model word vectors and lexicon scores for all strings in lexicon
    (to be used with sklearn models)
    :param model:
    :param lexicon:
    :return:
    """
    if sanitize:
        present = {k:lexicon[k] for k in lexicon.keys() if k in model.wv.vocab.keys()}
    else:
        present = lexicon
    
    return np.array([model.wv[w] for w in present.keys()]), np.array(list(present.values()))


def model_deltas(mod_a,mod_b):
    """
    Returns dictionary with word and delta-vector between model_a and model_b representations
    :param mod_a:
    :param mod_b:
    :return:
    """
    intersect = set(mod_a.wv.vocab.keys()).intersection(set(mod_b.wv.vocab.keys()))
    return { w: mod_a.wv[w] - mod_b.wv[w] for w in intersect}

def hotelling_lexicon(model, lexicon, alpha = 0.01):
    
    if len(lexicon.positive) < model.vector_size or len(lexicon.negative) < model.vector_size:
        raise RuntimeError("Lexicon cardianality too small for model: %s < %s" % 
                           (min(len(lexicon.positive),len(lexicon.negative)), model.vector_size ) )
    
    X1, y = parse_lexicon(model,lexicon.positive)
    X2, y = parse_lexicon(model,lexicon.negative)

    
    test = hotellings2(X1, X2)
    inference = test.inference(alpha)
    print("H0: Lexicon polarities have same means in model")
    print("H1: Lexicon polarities have different means in model")
    print("Reject H0 for H1: %s" %inference.h0reject)
    return inference.h0reject