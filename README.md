# MNLP-Assign-2
# Multilingual ASR

## Sub-Task 1

To train the model s2t_transformer_s from scratch using fairseq, we follow the following steps

```
pip install -r requirements.txt
```

Clone fairseq repo
```
git clone https://github.com/facebookresearch/fairseq.git
```
### Preprocessing
Download CommonVoice data from https://commonvoice.mozilla.org/en/datasets - We use Common Voice Corpus 15.0 for Subtask-1      
Place the data under /data/commonvoice/datasets/gn/ folder      

To preprocess the data according to CommonVoice format, replace the `examples/speech_to_text/prep_librispeech_data.py` by `prep_librispeech_data.py` in the repo

```
cd fairseq/
python examples/speech_to_text/prep_librispeech_data.py \
  --output-root /data/commonvoice/datasets/gn/ --vocab-type unigram --vocab-size 2000
```

### Training
To train run the following command

```
fairseq-train /data/commonvoice/datasets/gn/ --save-dir /data/commonvoice/datasets/gn/save \
  --config-yaml config.yaml --train-subset train --valid-subset dev \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8 --max-epoch 30 --log-format tqdm
```

### Prediction

LS_ROOT = '/data/commonvoice/datasets/gn/'
SAVE_DIR = '/data/commonvoice/datasets/gn/save'
CHECKPOINT_FILENAME = 'checkpoint_best.pt'
SUBSETS = ['test','dev']

# Loop over subsets
for SUBSET in SUBSETS:
    !fairseq-generate {LS_ROOT} --config-yaml config.yaml --gen-subset {SUBSET} \
    --task speech_to_text --path {SAVE_DIR}/{CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring wer
