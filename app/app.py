from flask import Flask, render_template, request
import numpy as np
import pickle
import torch
from models.classes import Skipgram, SkipgramNeg, Glove, CustomGensim

app = Flask(__name__)

# Loading Models

# Skipgram
skg_args = pickle.load(open('models/skipgrams.args', 'rb'))
model_skipgram = Skipgram(**skg_args)
model_skipgram.load_state_dict(torch.load('models/skipgram.model'))
# print(model_skipgram.word2index['happy'])

#SkipgramNeg
neg_args = pickle.load(open('models/neg.args', 'rb'))
model_neg = SkipgramNeg(**neg_args)
model_neg.load_state_dict(torch.load('models/neg.model'))

#Glove
glove_args = pickle.load(open('models/glove.args', 'rb'))
model_glove = Glove(**glove_args)
model_glove.load_state_dict(torch.load('models/glove.model'))

#Gensim
load_model = pickle.load(open('models/gensim.model', 'rb'))
model_gensim = CustomGensim(load_model)

models = {'skipgram': 'Skipgram', 'neg': 'SkipGramNeg', 'glove': 'Glove', 'gensim': 'Glove (Gensim)'}

# load corpus
import nltk
nltk.download('brown')

from nltk.corpus import brown

brown.categories()
corpus = brown.sents(categories="news")

def find_closest_indices_cosine(vector_list, single_vector, k=10):
    # Calculate cosine similarities
    similarities = np.dot(vector_list, single_vector) / (np.linalg.norm(vector_list, axis=1) * np.linalg.norm(single_vector))

    # Find indices of the top k closest vectors
    top_indices = np.argsort(similarities)[-k:][::-1]

    return top_indices

@app.route('/', methods=['GET'])
def index():
    # return '<h1>Hello from Flask & Docker</h2>'
    return render_template("index.html", models = models)

@app.route('/search', methods=['POST'])
def search():

    model_name = request.form['model']
    
    # {'skipgram': 'Skipgram', 'neg': 'SkipGramNeg', 'glove': 'Glove', 'gensim': 'Glove (Gensim)'}
    if model_name == 'skipgram':
        model = model_skipgram
    elif model_name == 'neg':
        model = model_neg
    elif model_name == 'glove':
        model = model_glove
    else:
        model = model_gensim

    query = request.form['query'].strip()

    # first computer sentence vector for a given query.
    qwords = query.split(" ")

    qwords_embeds = np.array([model.get_embed(word) for word in qwords])
    
    qsentence_embeds = np.mean(qwords_embeds, 0)

    corpus_embeds = []
    for each_sent in corpus:
        words_embeds = np.array([model.get_embed(word) for word in each_sent])
        sentence_embeds = np.mean(words_embeds, 0)
        corpus_embeds.append(sentence_embeds)

    corpus_embeds = np.array(corpus_embeds)

    result_idxs = find_closest_indices_cosine(corpus_embeds, qsentence_embeds)

    result = []
    for idx in result_idxs:
        result.append(' '.join(corpus[idx]))

    return render_template('index.html', models = models, result = result, model_name = model_name, old_query = query)


port_number = 8000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_number)