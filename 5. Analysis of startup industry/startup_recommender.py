#---------- IMPORTS ----------------#

import string
import flask
import pandas as pd
import numpy as np
import cPickle as pickle

from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import tokenize
from nltk.stem import PorterStemmer

#---------- MODEL IN MEMORY ----------------#

### load in combined and reduced matrix exported from IPython notebook
# reduced_matrix = np.load("svd2.npy")
# with open('combined.dat', 'rb') as infile:
#     combined = pickle.load(infile)
#     combined = sparse.csr_matrix(combined)


### load in attributes to populate results
# names - names of each company
# shortdescs - short descriptions of each company that are put into vectorizer
# hqs - headquarter of each company
# images - url to crunchbase image stores for each company
# categories - categories for each company
# raw_short - short description to display for each company
# funding - total funding received for each company
# homepages - homepage for each company
# founded - year founded for each company
# status - each company's current status

def get_attributes():
    with open("names.txt", "r") as f:
        names = f.read().split('\n')
        names = map(lambda x: x.decode('utf-8') if x else x, names)

    with open("shortdescs.txt", "r") as f:
        shortdescs = f.read().split('\n')
        shortdescs = map(lambda x: x.encode('utf-8') if x else x, shortdescs)

    with open("hq.txt", "r") as f:
        hqs = f.read().split('\n')
        hqs = map(lambda x: str(x).replace("nan", "Not available") if x else x, hqs)

    with open("images.txt", "r") as f:
        images = f.read().split('\n')
        images = map(lambda x: str(x) if x else x, images)

    with open("categories.txt", "r") as f:
        categories = f.read().split('\n')

    with open("raw_short.txt", "r") as f:
        descs = f.read().split('\n')
        descs = map(lambda x: str(x) if x else x, descs)

    def convert_funding(x):
        try:
            x = int(x)
            if x == 0:
                x = "Not available"
            else:
                x = "$" + "{:,}".format(x)
        except:
            x = "Not available"
        return x

    with open("total_funding.txt", "r") as f:
        funding = f.read().split('\n')
        funding = map(lambda x: convert_funding(x), funding)

    with open("homepages.txt", "r") as f:
        homepages = f.read().split('\n')
        homepages = map(lambda x: str(x) if x else x, homepages)

    with open("founded.txt", "r") as f:
        founded_on = f.read().split('\n')
        founded_on = map(lambda x: str(x) if x else x, founded_on)

    with open("status.txt", "r") as f:
        status = f.read().split('\n')
        status = map(lambda x: x.replace("operating", "Operating") \
                     .replace("acquired", "Acquired") \
                     .replace("ipo", "IPO") \
                     .replace("nan", "Unknown") if x else x, status)

    all_results = [names, shortdescs, hqs, images, categories, descs, funding, homepages, founded_on, status]

    return all_results


### call function and get results

all_results = get_attributes()
names = all_results[0]
shortdescs = all_results[1]
hqs = all_results[2]
images = all_results[3]
categories = all_results[4]
descs = all_results[5]
funding = all_results[6]
homepages = all_results[7]
founded = all_results[8]
status = all_results[9]

# def functions as arguments for vectorizers

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def get_stop_words():
    stop_words = stopwords.words("english")
    stop_words.append(["company", "founded", "firm", "800", "offers", "based", "usa", "offering", "inc"])
    for letter in string.ascii_lowercase:
        stop_words.append(letter)
    return stop_words

stop_words = get_stop_words()

# vectorize descriptions
tfidf = TfidfVectorizer(stop_words = stop_words, tokenizer = tokenize, ngram_range=(1, 2))
desc_vectors = tfidf.fit_transform(shortdescs)

# vectorize the categories
categories_tfidf = TfidfVectorizer(stop_words = "english")
cat_vectors = categories_tfidf.fit_transform(categories)

# combine the vectors
combined = sparse.hstack([desc_vectors, cat_vectors])

# reduce the into 500 components
trunc_algo = TruncatedSVD(n_components = 500)
trunc_vec = trunc_algo.fit_transform(combined)

# train Nearest Neighbors on reduced matrix
nbrs = NearestNeighbors(algorithm = "brute", metric = "cityblock", n_neighbors = 20)
nbrs.fit(trunc_vec)

# function for getting network
def get_neighbors(user_input):

    try:
        index = names.index(user_input)
        neighbors = nbrs.kneighbors(trunc_vec[index])[1][0]
        distances = nbrs.kneighbors(trunc_vec[index])[0][0]
        print "it worked"

    except:
        user_input_trans = user_input.split(",", 1)
        desc_input = user_input_trans[0].strip()
        cat_input = user_input_trans[1].strip()
        desc_trans = tfidf.transform([desc_input])
        categories_trans = categories_tfidf.transform([cat_input])
        new_vec = sparse.hstack([categories_trans, desc_trans])
        new_trunc_vec = trunc_algo.transform(new_vec)
        neighbors = nbrs.kneighbors(new_trunc_vec)[1][0]

    results_list = []

    for i in range(len(neighbors)):
        result_dict = {}

        result_dict["Rank"] = str(i)
        result_dict["Images"] = str(images[neighbors[i]])
        # result_dict["Distance"] = str(round(distances[i], 3))
        result_dict["Company"] = names[neighbors[i]]
        result_dict["Location"] = str(hqs[neighbors[i]])
        result_dict["Web"] = homepages[neighbors[i]]
        result_dict["Founded"] = str(founded[neighbors[i]])
        result_dict["Funding"] = funding[neighbors[i]]
        result_dict["Status"] = status[neighbors[i]]
        result_dict["Categories"] = categories[neighbors[i]]
        result_dict["Description"] = descs[neighbors[i]]

        results_list.append(result_dict)
    summary_stats = 0
    return [results_list,summary_stats]

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def home_page():
    """
    Homepage: serve our home page, index.html
    """
    with open("index.html", 'r') as home:
        return home.read()

# Get an example and return it's score from the predictor model
@app.route("/recommend", methods=["POST"])
def recommend():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    company_name = flask.request.json
    results = get_neighbors(company_name["company_name"])
    results2 = {"results": results[0], "summary":results[1]}
    return flask.jsonify(results2)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', debug=True)
