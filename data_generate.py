import numpy as np
from numpy import random

# save index for train and test
def split(path):
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
  ind = np.arange(0, len(tokens))
  random.shuffle(ind)
  shuffled_index = np.array([val for val in ind])
  np.savetxt('shuffled_index.txt', shuffled_index, fmt='%d')

def generate_data_seq2seq(path, train_write_path, test_write_path):
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
  shuffled_index = np.loadtxt('shuffled_index.txt', dtype=int)
  tokens = np.array(tokens)[shuffled_index].tolist()
  cutoff = int(.80 * len(tokens))
  training_sentences = tokens[:cutoff]
  test_sentences = tokens[cutoff:]
  # write
  with open(train_write_path, "w") as f:
    for elem in training_sentences:
      words = ""
      labels = ""
      for word, label in elem:
        words += word + " "
        labels += label + " "
      f.write(words[:-1] + "\t" + labels[:-1] + "\n")

  with open(test_write_path, "w") as f:
    for elem in test_sentences:
      words = ""
      labels = ""
      for word, label in elem:
        words += word + " "
        labels += label + " "
      f.write(words[:-1] + "\t" + labels[:-1] + "\n")

generate_data_seq2seq("./dataset/Indonesian_Manually_Tagged_Corpus.tsv", "./dataset/train.txt", "./dataset/test.txt")