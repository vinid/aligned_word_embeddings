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
* :code:`pip install git+https://github.com/valedica/gensim.git`
* cd in repository
* :code:`pip install -e .`

Note
----

* Remember: when you call the :code:`AlignedWordEmbeddings` the script creates a "model/" folder
* Remember: if a compass is present in the model folder, the model will not retrain a new compass

How To Use
----------

.. code-block:: python

    from aligned_word_embeddings import aligned_word_embeddings

    from gensim.models.word2vec import Word2Vec
    kd = aligned_word_embeddings.AlignedWordEmbeddings(size=30, siter=10, diter=10, workers=4)

    kd.train_static("compass_text")

    slice_one = kd.train_slice("slice_one", save=True)
    slice_two = kd.train_slice("slice_two", save=True)

    print(aligned_word_embeddings.most_similar_from_model("flat", silce_one, slice_two)) 

    slice_one = Word2Vec.load("model/slice_one.model") # model loading with gensim
    slice_two = Word2Vec.load("model/slice_one.model")



Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
