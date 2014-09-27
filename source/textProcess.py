import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def removeStopPunc(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
    stop = stopwords.words('english')
    newwords = [w for w in words if w not in stop]
    return newwords


def posTag(text):
    text = nltk.sent_tokenize(text)
    text = [nltk.word_tokenize(sent) for sent in text]
    text = [nltk.pos_tag(sent) for sent in text]
    return text


def posExtract(text, pattern):
    words = [word for sent in text for (word,tag) in sent if re.match(pattern, tag)]
    return words

