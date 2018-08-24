import os
import argparse
import logging

import torch
import torchtext

import seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.trainer import SupervisedTrainer

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', help='Give the checkpoint path from which to load the model')
parser.add_argument('--test_data', help='Path to test data')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)

opt = parser.parse_args()

if torch.cuda.is_available():
        print("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

#################################################################################
# load model

logging.info("loading checkpoint from {}".format(os.path.join(opt.checkpoint_path)))
checkpoint = Checkpoint.load(opt.checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

############################################################################
# Prepare dataset and loss
src = SourceField()
tgt = TargetField()
src.vocab = input_vocab
tgt.vocab = output_vocab
max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate test set
test = torchtext.data.TabularDataset(
    path=opt.test_data, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter
)

# Prepare loss
weight = torch.ones(len(output_vocab))
pad = output_vocab.stoi[tgt.pad_token]
loss = Perplexity(weight, pad)
if torch.cuda.is_available():
    loss.cuda()

#################################################################################
# Evaluate model on test set

evaluator = Evaluator(loss=loss, batch_size=opt.batch_size)
losses, accuracy, totals = evaluator.evaluate(seq2seq, test)

prec_tot = 0
rec_tot = 0
f1_tot = 0
for key in totals:
  precision = totals[key]['tp'] / (totals[key]['tp'] + totals[key]['fp'])
  prec_tot += precision
  recall = totals[key]['tp'] / (totals[key]['tp'] + totals[key]['fn'])
  rec_tot += recall
  f1 = 2 * (precision * recall) / (precision + recall)
  f1_tot += f1
  # print(key + " precision: " + str(precision) + " recall: " + str(recall) + " f1: " + str(f1))
  print(key + " & " + str("{0:.3f}".format(round(precision,2))) + " & " + str("{0:.3f}".format(round(recall,2)))
        + " & " + str("{0:.3f}".format(round(f1,2))) + " \\\\")
# print("Total precision: " + str(prec_tot / len(totals)) + " recall: " + str(rec_tot / len(totals)) +
#       " f1: " + str(f1_tot/ len(totals)))
print("Total & " + str("{0:.3f}".format(round(prec_tot / len(totals),2))) + " & "
      + str("{0:.3f}".format(round(rec_tot / len(totals),2))) +
      " & " + str("{0:.3f}".format(round(f1_tot/ len(totals),2))) + " \\\\")
# total_loss, log_msg, _ = SupervisedTrainer.print_eval(losses, metrics, 0)
#
# print(log_msg)