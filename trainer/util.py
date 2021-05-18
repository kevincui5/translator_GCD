import numpy as np
import tensorflow as tf
#import keras.backend as K
#from pickle import load
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from numpy import array
from numpy import genfromtxt
'''
def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - tf.math.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
'''
# load a clean dataset
def load_clean_sentences(filename):
    return genfromtxt(filename, delimiter=',', dtype=str)
	#return load(open(filename, 'rb'))

def load_csv_batch(filename, start_row, rows):
    return genfromtxt(filename, delimiter=',', dtype=str, 
                      skip_header = start_row, max_rows = rows)
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(inp_tokenizer, length, lines):
	# integer encode sequences
	X = inp_tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tar_tokenizer, source):
    prediction = model.predict(source, verbose=0)
    #if prediction has extra outmost dimension of 1, need to remove it
    prediction = np.squeeze(prediction, axis=0)
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tar_tokenizer)
        if word is None:
            continue
        target.append(word)
    return ' '.join(target)

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	#dump(sentences, open(filename, 'wb'))
    np.savetxt(filename, sentences, delimiter=",", fmt='%s')
    print('Saved: %s' % filename)

#batch generator for keras model
#reads dataset on disk in batches
