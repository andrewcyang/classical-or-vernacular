#!/usr/bin/env python
# coding: utf-8

# Classifying classical and vernacular Chinese text with a random forest
#
# @author Andrew Yang

import numpy as np
import pandas as pd

train_set = pd.read_csv('./data/train.txt')
unlabeled_set = pd.read_csv('./data/test.txt')

train_set.head(10)

train_set.drop('id', axis=1, inplace=True)
unlabeled_set.drop('id', axis=1, inplace=True)

train_set['text'] = train_set['text'].str.replace(' ', '')
unlabeled_set['text'] = unlabeled_set['text'].str.replace(' ', '')

ind_vernacular = train_set['y'] == 0
ind_classical  = train_set['y'] == 1

vernacular = train_set[ind_vernacular]
classical  = train_set[ind_classical]


from collections import Counter

st_names, ind = ['vernacular', 'classical'], 0
for st in [vernacular, classical]:
    cnt = Counter()
    
    for inst in range(len(st)):
        ln = st.iloc[inst, 0]
        
        for char in ln:
            cnt[char] += 1
            
        com = cnt.most_common(10)
        
    print('Most common', st_names[ind], 'characters:')
    
    for i, c in enumerate(com):
        print(str(i+1)+'.', c[0]+':', c[1], 'instances')
        
    print()
    ind += 1

from gensim.models import Word2Vec
model = Word2Vec(sentences=train_set['text']) # Leaving other parameters as default

chars_list = [['矣'], model.wv.most_similar('矣', topn=10), ['天'], model.wv.most_similar('天', topn=10)]

for chars in chars_list:
    out = ''
    
    for char in chars:
        out += char[0]
    print(out)
    
    if len(char) != 1:
        print()

m = len(train_set) + len(unlabeled_set)
dims = 100

vector = np.zeros([m, dims])

text = list(train_set['text']) + list(unlabeled_set['text'])
for ind, ln in enumerate(text):
    cnts, row = 0, 0
    
    for char in ln:
        try:
            row += model.wv[char]
            cnts += 1
            
        except:
            pass
        
    vector[ind, :] = row / cnts

from sklearn.model_selection import train_test_split
A = vector[:len(train_set)]
B = np.transpose(np.array([train_set['y']]))

vectorized_mat = np.concatenate([A, B], axis=1)
n = len(vectorized_mat[0])

X = vectorized_mat[:,:n-1]
y = vectorized_mat[:,n-1]

X_unlabeled = vector[len(train_set):]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

train_acc = []
test_acc = []

depth = range(3, 11)
for n_depth in depth:
    forest = RandomForestClassifier(n_estimators=100, max_depth=n_depth, n_jobs=-1, random_state=0)
    forest.fit(X_train, y_train)
    
    train_acc.append(forest.score(X_train, y_train))
    test_acc.append(forest.score(X_test, y_test))

import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('seaborn')

plt.title('Training and testing accuracies vs. Max depth')
plt.plot(
         depth, 
         train_acc, 
         label='training accuracies', 
         color='green'
)
plt.plot(
         depth, 
         test_acc, 
         label='testing accuracies', 
         color='blue'
)
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


forest = RandomForestClassifier(n_estimators=100, max_depth=7, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)

pred = forest.predict(X_unlabeled)
pred = pd.DataFrame(pred, columns=['prediction'])

pred.replace(to_replace=0.0, value='vernacular', inplace=True)
pred.replace(to_replace=1.0, value='classical', inplace=True)

final = pd.concat([unlabeled_set, pred], axis=1)
final.head()

X_train_predict = forest.predict(X_train)
errors = X_train_predict != y_train
error_lines = X_train[errors]


for i in range(5):
    for ind in range(0, len(vector)):
        if np.array_equal(error_lines[i], vector[ind]):
            if train_set.iloc[ind,1] == 0:
                print('Actual: Vernacular')
            else:
                print('Actual: Classical')
            print(train_set.iloc[ind,0])
            print()
