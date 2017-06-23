# crnn
Fork from https://github.com/ultimate010/crnn


### Installation
Full [anaconda2](https://www.continuum.io/downloads) package 
Theano
Keras

```
conda install theano pygpu
pip install keras
```

Switch Keras backend to Theano ([How-to](https://keras.io/backend/)) 



Preprocess data

```
python process_sst2_data.py w2v_bin sst_dir
```


To run on GPU

```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python sst2_cnn_rnn
```