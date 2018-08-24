# indonesian-pos-tag

## CRF 

CRF is implemented using crfsuite from scikit-learn

Parameters for crf-nltk.py:

* --train or --test

* --model_path

training

      python crf-nltk.py --train --model_path $PATH_TO_MODEL
      
evaluating
      
      python crf-nltk.py --test --model_path $PATH_TO_MODEL

## Seq2seq

Seq2seq implementation is built on top of IBM-seq2seq-pytorch library.

Sample usage for seq2seq.py:

training

     python examples/seq2seq.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
     
resuming from the latest checkpoint of the experiment
     
     python examples/seq2seq.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume

perform labeling with the chosen checkpoint and create the confusion matrix
      
      python examples/seq2seq.py --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR
      
Sample usage for evaluate.py:

      python --test_data dataset/test.txt --checkpoint_path experiment/checkpoints/2018_08_22_17_04_19


