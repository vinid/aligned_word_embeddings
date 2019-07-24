
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
