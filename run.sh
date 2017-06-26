# preprocess
python process_sst2_data.py /media/vdvinh/25A1FEDE380BDADA/data/GoogleNews-vectors-negative300.bin data/stanfordSentimentTreebank

# run
THEANO_FLAGS=mode=FAST_RUN,device=cuda0,floatX=float32 python sst2_cnn_rnn.py

THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python sst2_cnn_rnn_keras1.py