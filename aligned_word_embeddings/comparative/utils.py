import itertools
import numpy as np
from numpy.linalg import norm
from spm1d.stats import hotellings2
import pickle
import pandas as pd
import itertools
from collections import Counter
import codecs

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
        for a,b in itertools.combinations(aligned.trained_slices.items(),2):
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
        for a,b in itertools.combinations(aligned.trained_slices.items(),2):
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

def lexicon_polarity(model, lexicon, alpha = 0.05, verbose=True):
    """
    Tests equality of lexicon polarity means according to embedding model.
    Returns statistical test outcome against null hypothesis of equal means at given alpha value.
    :param model:
    :param lexicon:
    :param alpha:
    :param verbose:
    :return:
    """
    if len(lexicon.positive) < model.vector_size or len(lexicon.negative) < model.vector_size:
        raise RuntimeError("Lexicon cardianality too small for model: %s < %s" %
                           (min(len(lexicon.positive),len(lexicon.negative)), model.vector_size ) )

    X1, y = parse_lexicon(model,lexicon.positive)
    X2, y = parse_lexicon(model,lexicon.negative)


    test = hotellings2(X1, X2)
    inference = test.inference(alpha)
    if verbose: print("H0: Lexicon polarities have same means in model")
    if verbose: print("H1: Lexicon polarities have different means in model")
    if verbose: print("Reject H0 for H1: %s" %inference.h0reject)
    return inference.h0reject

def lexicon_stability(aligned, lexicon, mode='s2c', alpha=0.05, verbose=True):
    #WIP
    """
    Tests equality of lexicon polarity means during alignement.
    Two operation mode: slice-vs-slice (s2s) or slice-vs-compass (s2c)
    Returns statistical test outcome against null hypothesis of equal means at given alpha value.
    :param aligned:
    :param lexicon:
    :param mode:
    :param alpha:
    :param verbose:
    :return:
    """

    if len(lexicon.positive) < aligned.compass.vector_size or len(lexicon.negative) < aligned.compass.vector_size:
        raise RuntimeError("Lexicon cardianality too small for model: %s < %s" %
                           (min(len(lexicon.positive),len(lexicon.negative)), aligned.compass.vector_size ) )

    if mode not in ["s2s", "s2c"]:
        raise RuntimeError("Unkown mode selected, must be either 's2s' or 's2c'")
    else:
        results = {}
        if mode == "s2c":

            if verbose:
                print("Slice-vs-Compass mode selected")
                print("Detected %s slices" % len(aligned.trained_slices))


            XCP, y = parse_lexicon(aligned.compass,lexicon.positive)
            XCN, y = parse_lexicon(aligned.compass,lexicon.negative)

            for n,s in aligned.trained_slices.items():

                XSP, y = parse_lexicon(s,lexicon.positive)
                XSN, y = parse_lexicon(s,lexicon.negative)

                testP = hotellings2(XCP, XSP)
                testN = hotellings2(XCN, XSN)

                inferenceP = testP.inference(alpha)
                inferenceN = testN.inference(alpha)

                results[n+"_pos"] = inferenceP.h0reject
                results[n+"_neg"] = inferenceN.h0reject

                if verbose:
                    print(n,"H0: Positive Lexicon has same means")
                    print(n,"H1: Positive Lexicon has different means")
                    print(n,"Reject H0 for H1: %s" %inferenceP.h0reject)
                    print()
                    print(n,"H0: Negative Lexicon has same means")
                    print(n,"H1: Negative Lexicon has different means")
                    print(n,"Reject H0 for H1: %s" %inferenceN.h0reject)
                    print()

        if mode == "s2s":
            if verbose:
                print("Slice-vs-Slice mode selected")
                print("Detected %s slices" % len(aligned.trained_slices))


            for a,b in itertools.combinations(aligned.trained_slices.items(),2):

                n = a[0]+"-"+b[0]

                XAP, y = parse_lexicon(a[1],lexicon.positive)
                XAN, y = parse_lexicon(a[1],lexicon.negative)

                XBP, y = parse_lexicon(b[1],lexicon.positive)
                XBN, y = parse_lexicon(b[1],lexicon.negative)

                testP = hotellings2(XAP, XBP)
                testN = hotellings2(XAN, XBN)

                inferenceP = testP.inference(alpha)
                inferenceN = testN.inference(alpha)

                results[n+"_pos"] = inferenceP.h0reject
                results[n+"_neg"] = inferenceN.h0reject

                if verbose:
                    print(n,"H0: Positive Lexicon has same means")
                    print(n,"H1: Positive Lexicon has different means")
                    print(n,"Reject H0 for H1: %s" %inferenceP.h0reject)
                    print()
                    print(n,"H0: Negative Lexicon has same means")
                    print(n,"H1: Negative Lexicon has different means")
                    print(n,"Reject H0 for H1: %s" %inferenceN.h0reject)
                    print()

    return results


def dump_awec(dumping_path, awec_object):
    """
    saves an awec object using pickle
    :param dumping_path:
    :param awec_object:
    :return:
    """
    with open(dumping_path, 'wb') as f:
        pickle.dump(awec_object, f)


def load_awec(loading_path):
    """
    loads an awec object using pickle
    :param loading_path:
    :return:
    """
    with open(loading_path, 'rb') as f:
        return pickle.load(f)


def zipf(word,wordcounts,d1=None,d2=None):
    '''
    Compute zipf frequency measure for given work w.r.t given corpus wordcounts
    :param word: string
    :param wordcounts: dictionary with token occurrencies in corpus (e.g. collections.Counter)
    :param d1: float with total number of token in corpus in millions
    :param d2: float with number of unique tokes in corpus in millions
    '''
    n = (wordcounts[word] + 1)
    if d1 is None:
        d1 = sum(wordcounts.values())/1000000
    if d2 is None:
        d2 = len(wordcounts.keys())/1000000

    return np.log10( n/(d1+d2) ) + 3

def corresp_transpose(word, mod_a, mod_b, reverse=False):
    if reverse:
        mod_a, mod_b = mod_b ,mod_a
    try:
        assert word in mod_a.wv.vocab.keys()
    except:
        return None
    return mod_b.wv.most_similar([mod_a.wv[word] ], topn=1 )[0][0]

def augment_single(word, model, n=1, m=1000, sample=True):
    '''Data augmentation inside single model'''
    synth = []
    seed = model.wv[word]
    for _ in range(n):
        p = []
        curr = seed
        for _ in range(m):
            step = np.random.normal(size=(model.vector_size))
            step /= norm(step)

            curr = curr + step

            if model.wv.most_similar([curr], topn=1)[0][0] == word:
                p.append(curr)
            else:
                break
        if sample: p = p[np.random.choice(range(len(p)))]
        synth.append(p)
    return np.array(synth)


def augment_multi(word, models, n=1, m=1000, sample=True):
    '''Data augmentation across multiple models'''
    synth = []
    seed = np.mean(np.vstack( [mods.wv[word] for mods in models]),axis=0)
    for _ in range(n):
        p = []
        curr = seed
        for _ in range(m):
            step = np.random.normal(size=(models[0].vector_size))
            step /= norm(step)

            curr = curr + step

            if all([mods.wv.most_similar([curr], topn=1)[0][0] == word for mods in models]):
                p.append(curr)
            else:
                break
        if sample: p = p[np.random.choice(range(len(p)))]
        synth.append(p)

    return np.array(synth)


def vocabulary_dataframe(corpora, verbose=False, filter_common = True, min_count=5, encoding="utf-8"):
    '''Build zipf dataframe for corpora'''
    # build zipf dicts
    wordzipfers = {}
    for f in corpora:
        name = f.split("/")[-1][:-4]
        with codecs.open(f, encoding=encoding) as fin:
            wc = Counter(fin.read().split())
        if verbose: print("count",name)
        
        # filter on min_count frequency
        if min_count:
            wc = {x : wc[x] for x in wc if wc[x] >= min_count}
            if verbose: print("min_count",name)

        
        d1 = sum(wc.values())/1000000
        d2 = len(wc.keys())/1000000
        wordzipfers[name] = { word:zipf(word,wc,d1,d2) for word in wc.keys()}
        if verbose: print("zipf",name)
        del wc
        
    # build dataframe of zipfs
    df = pd.DataFrame({"zipf %s" % (k) :v for k,v in wordzipfers.items()})

    if verbose: print("dataframe done")    
    
    # filter on shared vocab
    if filter_common:
        shared = None
        for k in wordzipfers.keys():
            if not shared:
                shared = set(wordzipfers[k].keys())
            else:
                shared = shared.intersection(set(wordzipfers[k].keys()))
        df = df.filter(shared,axis=0)
        if verbose: print("common filter done")
    
    return df


def comparative_dataframe(models, corpora, min_count = 5, filter_common = True, verbose = True, encoding="utf-8"):
    ''' Build comparative dataframe for multiple models'''
    names = [f.split("/")[-1][:-4] for f in corpora]

    if len(models)!=len(corpora):
        raise RuntimeError("models and corpora must be same length")

    df = vocabulary_dataframe(corpora, verbose=verbose, filter_common = filter_common, min_count=min_count, encoding=encoding)

    for i, j in itertools.permutations(range(len(models)),2):
        df['Corr %s2%s' %(names[i],names[j])] = [ corresp_transpose(x,models[i],models[j]) for x in df.index ]
        df['Corr %s2%s' %(names[j],names[i])] = [ corresp_transpose(x, models[i],models[j],reverse=True) for x in df.index ]

    return df
