
from tika import parser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from num2words import num2words
import re
import string
import numpy as np

def pdftotext_converter(filename):
    raw = parser.from_file(filename)
    return(raw['content'])


def cleanText(pdf_file_text):
    # Convert all strings to lowercase

    pdf_file_text = pdf_file_text.lower()

    #  Remove numbers, punctuation, empty lines, etc.....
    pdf_file_text = re.sub(r'\d+', '', pdf_file_text)

    pdf_file_text = pdf_file_text.translate(str.maketrans('', '', string.punctuation))

    pdf_file_lines = pdf_file_text.split("\n")
    non_empty_lines = [line for line in pdf_file_lines if line.strip() != ""]
    pdf_file_without_empty_lines = ""
    for line in non_empty_lines:
        pdf_file_without_empty_lines += line + "\n"

    pdf_file_without_empty_lines = pdf_file_without_empty_lines.rstrip("\n")

    return pdf_file_without_empty_lines


def convert_lower_case(data):
    return np.char.lower(data)
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
def stemming(data):
    stemmer = PorterStemmer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
def preprocess(data):
    longword = re.compile(r'\W*\b\w{20,}\b')
    data=longword.sub('', data)
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data
