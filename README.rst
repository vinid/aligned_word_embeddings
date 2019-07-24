=======================
Aligned Word Embeddings
=======================


.. image:: https://img.shields.io/pypi/v/aligned_word_embeddings.svg
        :target: https://pypi.python.org/pypi/aligned_word_embeddings

.. image:: https://img.shields.io/travis/vinid/aligned_word_embeddings.svg
        :target: https://travis-ci.org/vinid/aligned_word_embeddings

.. image:: https://readthedocs.org/projects/aligned-word-embeddings/badge/?version=latest
        :target: https://aligned-word-embeddings.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




* Free software: GNU General Public License v3
* Documentation: https://aligned-word-embeddings.readthedocs.io.


Installing
----------

* Clone the repository
* :code:`virtualenv -p python3.6 env`
* :code:`source env/bin/activate`
* :code:`pip install cython`
* :code:`pip install git+https://github.com/valedica/gensim.git`
* cd in repository
* :code:`pip install -e .`

Note
----

* Remember: when you call the :code:`AlignedWordEmbeddings` the script creates a "model/" folder

How To Use
----------

* Train

.. code-block:: python

    from aligned_word_embeddings.aligned_word_embeddings import  AlignedWordEmbeddings
    from aligned_word_embeddings.spatial.distances import most_similar_from_model
    from gensim.models.word2vec import Word2Vec

    aligner = AlignedWordEmbeddings(size=30, siter=10, diter=10, workers=4)

    aligner.train_compass("/path/to/compass_text", overwrite=False) # keep an eye on the overwrite behaviour

    slice_one = aligner.train_slice("/path/to/slice_one.txt", save=True)
    slice_two = aligner.train_slice("/path/to/slice_two.txt", save=True)

    print(most_similar_from_model("flat", silce_one, slice_two))

    slice_one = Word2Vec.load("model/slice_one.model") # model loading with gensim
    slice_two = Word2Vec.load("model/slice_one.model")




..

* Load and Visualize

.. code-block:: python

    from aligned_word_embeddings.utils.visualization import visualize_embeddings

    model1 = Word2Vec.load("slice_one.model")
    model2 = Word2Vec.load("slice_two.model")

    import numpy as np
    models = [model1, model2]

    names = ["slice_one", "slice_two"]

    key = "word"

    visualize_embeddings(key, models, names, topn=2)

..

* Moral Embeddings

.. code-block::python

    from aligned_word_embeddings.datasets import moral
    from sklearn import linear_model
    from sklearn import preprocessing
    import numpy as np

    md = moral.MFDataset("MFD2.0.dic")

    X, y = md.get_moral(slice_one, "care")


    model = linear_model.LinearRegression()
    model.fit(X, y)

    def test_some_words_in_model(words, model):
        print("---------")
        for word in words:
            vv = preprocessing.normalize(np.array([slice_one.wv[word]]), norm="l2")
            vv = vv[0]
            print(word, model.predict(np.array([vv]))[0])
            print("---------")

    test_some_words_in_model(["boris"], model)
..

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
