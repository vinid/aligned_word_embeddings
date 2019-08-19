from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


class TextCleaner:
    """
    Class cleaning utility. Word tokenization with stopwords and non alpha chars removal
    """
    def __init__(self, language):
        self.language = language
        self.stopwords = stopwords.words(self.language)

    def cleaning_text(self, text):
        cleaning_text  = word_tokenize(text)

        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in cleaning_text]

        words = [word.lower() for word in stripped if word.isalpha()]
        words = [w for w in words if not w in self.stopwords]
        cleaning_text = " ".join(words)
        return cleaning_text
