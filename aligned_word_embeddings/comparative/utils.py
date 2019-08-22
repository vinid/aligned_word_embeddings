import itertools

_MEASURES = ["jaccard", "count", "raw"]

def vocabulary_overlap(mod_a, mod_b, measure="jaccard"):
    
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
    tot_a = sum([x.count for x in mod_a.wv.vocab.values()])
    tot_b = sum([x.count for x in mod_b.wv.vocab.values()])
    return tot_a / tot_b

def frequency_report(aligned, internal=False):
    if internal:
        for a,b in itertools.combinations(aligned.trained_slices.items()):
            print("%s / %s: " % (a[0],b[0]), frequency_rate(a[1],b[1]) )
        return
    else:
        for n,s in aligned.trained_slices.items():
            print("%s / compass: " % n,  frequency_rate(s,aligned.compass) )
        return