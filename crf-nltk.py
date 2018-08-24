import nltk
import numpy as np
from numpy import random
import torch
from torch.autograd import Variable
from sklearn.externals import joblib
from sklearn_crfsuite import CRF
from nltk.tag.util import untag
from sklearn_crfsuite import metrics

import argparse
from sklearn.metrics import confusion_matrix
from helper import print_confusion_matrix
import collections
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.set_defaults(train=True)
parser.add_argument('--model_path', help='Path to load or save model')

opt = parser.parse_args()

# preprocess
def read_data(path):
  tokens = []
  with open(path, "r") as f:
    elem = []
    for line in f:
      ar = line.split("\t")
      if len(ar) == 1:
        tokens.append(elem)
        elem = []
      else:
        tok = ar[0]
        lab = ar[1].replace("\n", "").upper()
        elem.append((tok, lab))
  # ind = np.arange(0, len(tokens))
  # random.shuffle(ind)
  # shuffled_index = np.array([val for val in ind])
  # np.savetxt('shuffled_index.txt', shuffled_index, fmt='%d')
  shuffled_index = np.loadtxt('shuffled_index.txt', dtype=int)
  tokens = np.array(tokens)[shuffled_index].tolist()
  return tokens

tagged_sentences = read_data("./dataset/Indonesian_Manually_Tagged_Corpus.tsv")

# Split the dataset for training and testing
cutoff = int(.80 * len(tagged_sentences))
training_sentences = tagged_sentences[:cutoff]
test_sentences = tagged_sentences[cutoff:]


def features(sentence, index):
  """ sentence: [w1, w2, ...], index: the index of the word """
  return {
    # word features
    'word': sentence[index],
    'is_first': index == 0,
    'is_last': index == len(sentence) - 1,
    'is_capitalized': sentence[index][0].upper() == sentence[index][0],
    'is_all_caps': sentence[index].upper() == sentence[index],
    'is_all_lower': sentence[index].lower() == sentence[index],
    # prefixes and suffixes
    'prefix-1': sentence[index][0],
    'prefix-2': sentence[index][:2],
    'prefix-3': sentence[index][:3],
    'suffix-1': sentence[index][-1],
    'suffix-2': sentence[index][-2:],
    'suffix-3': sentence[index][-3:],
    # neighboring words
    'prev_word': '' if index == 0 else sentence[index - 1],
    'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    # word features 2
    'has_hyphen': '-' in sentence[index],
    'is_numeric': sentence[index].isdigit(),
    'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
  }

def transform_to_dataset(tagged_sentences):
  X, y = [], []

  for tagged in tagged_sentences:
    X.append([features(untag(tagged), index) for index in range(len(tagged))])
    y.append([tag for _, tag in tagged])

  return X, y


X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(test_sentences)

# print(len(X_train))
# print(len(X_test))
# print(X_train[0])

# for el in training_sentences:
#   print(el)

filename = opt.model_path
train = opt.train

if train:
  model = CRF()
  model.fit(X_train, y_train)

  joblib.dump(model, filename)

else:
  model = joblib.load(filename)
  sentence = ["Siapa", "nama", "aku", "yang", "ganteng", "ini", "?"]

  def pos_tag(sentence):
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))

  analysis = collections.defaultdict(lambda: collections.defaultdict(int))
  tagss = []
  tagss_pred = []
  for elem in test_sentences:
    sentence = []
    tags = []
    for w in elem:
      sentence.append(w[0])
      tags.append(w[1])
    #print(sentence)
    preds = pos_tag(sentence)
    tags_pred = [x[1] for x in preds]
    check = False
    for a, b in zip(tags, tags_pred):
      analysis[a][b] += 1
      if a != b:
        check = True
    tagss.append(tags)
    tagss_pred.append(tags_pred)
    if check:
      pr_str = ""
      for a, b, c in zip(sentence, tags, tags_pred):
        pr_str += a + "/" + b + "/" + c + " "
      #print(pr_str[:-1])

  labels = list(model.classes_)
  sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
  )
  print_confusion_matrix(analysis, sorted_labels)

  # show statistics
  y_pred = model.predict(X_test)
  a = metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
  )
  # print(a)

  # # confusion matrix
  # y_test_flat = [item for sublist in y_test for item in sublist]
  # y_pred_flat = [item for sublist in y_pred for item in sublist]
  # cm = confusion_matrix(y_test_flat, y_pred_flat)
  # #print(sorted_labels)
  # print_confusion_matrix(cm, sorted_labels)