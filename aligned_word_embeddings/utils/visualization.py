import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def visualize_embeddings(key, models, names, topn=5):
    words_in_models = []
    embeddings = []
    for index, model in enumerate(models):
        print(index)
        words_in_model = []
        # finding words-embeddings
        words_in_model.append(key + "_" + names[index])
        embeddings.append(model.wv[key])

        local_words = [k[0] for k in model.wv.most_similar(key, topn=30)]
        local_words = list(filter(lambda x : x[0:4] == "dbr:", local_words))[0:topn]

        for similar_word in local_words:

            words_in_model.append(similar_word)
            embeddings.append(model.wv[similar_word])
        words_in_models.append(words_in_model)

    embedding_clusters = np.array(embeddings)
    pca_object = PCA(n_components=2)
    embeddings_2d = np.array(pca_object.fit_transform(embedding_clusters))

    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    for index, model in enumerate(models):
        for index_word, word in enumerate(words_in_models[index]):
            x = embeddings_2d[index_word+(topn+1)*index][0]
            y = embeddings_2d[index_word+(topn+1)*index][1]
            plt.scatter(x, y, c=colors[index], alpha=0.5, label=word)

            plt.annotate(word, alpha=0.5, xy=(x, y), xytext=(5, 2),
                             textcoords='offset points', ha='right', va='bottom', size=8)


    plt.legend(loc=4)
    plt.title("Multiple Spaces Plot")
    plt.grid(True)
    plt.show()
