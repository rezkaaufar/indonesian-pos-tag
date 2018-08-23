# indonesian-pos-tag

## CRF 

Parameters for crf-nltk.py:

* --train or --test

* --model_path

      python crf-nltk.py --train --model_path $PATH_TO_MODEL

## Deep Learning

Sample usage for seq2seq.py:

training

     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
     
resuming from the latest checkpoint of the experiment
     
     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume

resuming from a specific checkpoint
      
      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR


