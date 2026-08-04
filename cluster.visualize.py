# -*- coding: utf-8 -*-
### INFO: in order to install and use gensim with the pretrained model of
#   2018, you should install gensim 3.8, so you should use Python 3.8.0 and pip from Python 3.8.0 + pip install wheel; use alias python=python3.8 before using python
### USAGE: python cluster.visualize.py
##python3.8 -m pip install seaborn


import sys
import gensim, logging
from pprint import pprint as print
from gensim.models.fasttext import FastText
from gensim.test.utils import datapath
#from whatlies.language import SpacyLanguage #FasttextLanguage #pip install whatlies
import matplotlib.pylab as plt 
#import plotly.express as px
#from whatlies import Embedding
import numpy as np
from gensim.models.fasttext import FastText
#from gensim.scripts.word2vec2tensor import word2vec2tensor
# Taken from: https://www.kdnuggets.com/2018/04/robust-word2vec-models-gensim.html
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import plotly
import plotly.express as px
import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.KeyedVectors.load('araneum_none_fasttextcbow_300_5_2018.model')

with open('clusters/impf.pfv.vrb.txt', encoding='utf-8') as f:
    words = f.read().splitlines()

with open('clusters/impf.pfv.label.txt', encoding='utf-8') as f:
    asp = f.read().splitlines()
print(asp)    
#words = ["делать", "переделать", "доносить", "сносить", "уносить", "доделать","приделать", "сделать", "уделать", "наделать"]
emb = model[words]
print(emb)

vec_df = pd.DataFrame(emb)
vec_df.loc[:, "vrbs"] = words
vec_df.loc[:, "asp"] = asp
#print(vec_df)

labels = asp
tsne = TSNE(n_components = 2, random_state = 0, n_iter = 10000, perplexity = 40)
tsne_results = tsne.fit_transform(emb)
print(tsne_results)
tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
tsne_results.loc[:, "asp"] = asp
#color_dict = {'на':'red', 'по':'blue', 'раз':'grey', 'от':'green','раз':'purple'}

print(tsne_results)

sns.scatterplot(data=tsne_results, x='tsne1', y='tsne2', hue='asp', palette="hls")
plt.legend(loc='upper right')
plt.show()
