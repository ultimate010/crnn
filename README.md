# crnn
Fork from https://github.com/ultimate010/crnn


### Installation
Full [anaconda2](https://www.continuum.io/downloads) package 
Theano
Keras (install version 1.2.2 for reproduce acc > 89 %)

```
conda install theano pygpu
pip install keras=1.2.2
```

Switch Keras backend to Theano ([How-to](https://keras.io/backend/)) 

Download word2vec pretrained binary file [here](https://code.google.com/archive/p/word2vec/) 
(looking for GoogleNews-vectors-negative300.bin.gz)

Preprocess data

```
python process_sst2_data.py w2v_bin sst_dir
```


To run on GPU

```
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python sst2_cnn_rnn_keras1.py
```

