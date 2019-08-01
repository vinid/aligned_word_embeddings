from scipy.spatial.distance import cosine

def most_similar_from_model(word, model_start, model_to, topn=10):
    """
    Returns the most similar words' vectors in model_to of a word vector in model_start
    :param word:
    :param model_start:
    :param model_to:
    :param topn:
    :return:
    """
    vector = model_start[word]
    return model_to.most_similar([vector], topn=topn)


def distance_from_third(third, model1, model2, word):
    """
    How distant are the models from a third model (e.g, from the compass)
    :param third:
    :param model1:
    :param model2:
    :param word:
    :return:
    """
    first = aligned_similarity(third, model1, word)
    second = aligned_similarity(third, model2, word)

    if first > second:
        return -1, (first - second)
    elif first < second:
        return 1, (second - first)
    else:
        return 0, 0


def aligned_similarity(model1, model2, word):
    """
    Cosine similarity
    :param model1:
    :param model2:
    :param word:
    :return:
    """
    return 1 - cosine(model1.wv[word], model2.wv[word])
