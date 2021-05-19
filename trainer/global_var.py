from trainer.util import *
import json
from datetime import datetime
import numpy as np
from tensorflow.data import Dataset


def load_data_helper(filename, idx, batch_size, n_s, params):
    data_raw = load_csv_batch(filename, idx*batch_size, batch_size)
    json_log = open('debug_data_log.json', mode='at', buffering=1)
    json_log.write(
        json.dumps({'time': datetime.now().strftime("%H:%M:%S%f"), 
                    'data batch#': idx,
                    'data_raw': np.array2string(data_raw)}) + '\n')
    json_log.close()
    
    eng_tokenizer=params.get('eng_tokenizer')
    ger_tokenizer=params.get('ger_tokenizer')
    eng_length=params.get('eng_length')
    ger_length=params.get('ger_length')
    ger_vocab_size=params.get('ger_vocab_size')
    eng_vocab_size=params.get('eng_vocab_size')
    X = encode_sequences(ger_tokenizer, ger_length, data_raw[:, 1])
    Y = encode_sequences(eng_tokenizer, eng_length, data_raw[:, 0])
    Y_oh = encode_output(Y, eng_vocab_size) #converted to one hot

    inputs = [X,np.zeros((X.shape[0],n_s)),np.zeros((X.shape[0],n_s)),np.zeros((X.shape[0], 1))]
    outputs = Y_oh
    return inputs, outputs, data_raw

#raw data is the two column sentences, one for input, the other for out   
def get_raw_data_tokenizer(csv_path):
    raw_data = load_clean_sentences(csv_path)
    #raw_data = pd.read_csv(csv_path, sep=',').values
    #raw_data = raw_data.values
    return get_tokenizer_helper(raw_data)

def get_tokenizer_helper(raw_data):
    ger_tokenizer = create_tokenizer(raw_data[:, 1])
    eng_tokenizer = create_tokenizer(raw_data[:, 0])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    ger_length = max_length(raw_data[:, 1])
    eng_length = max_length(raw_data[:, 0])
    result = {'ger_tokenizer':ger_tokenizer,
              'eng_tokenizer':eng_tokenizer,
              'ger_vocab_size':ger_vocab_size,
              'eng_vocab_size':eng_vocab_size,
              'ger_length':ger_length,
              'eng_length':eng_length}
    return result

def create_input_output_tensor(raw_data, params, n_s):
    X = encode_sequences(params.get('ger_tokenizer'), params.get('ger_length'), raw_data[:, 1]) #(m,Tx)
    Y = encode_sequences(params.get('eng_tokenizer'), params.get('eng_length'), raw_data[:, 0]) #(m,Ty)
    Y_oh = encode_output(Y, params.get('eng_vocab_size')) #converted to one hot (m,Ty,vocab_tar_size)
    inputs = {"X": X,
                    "s0": np.zeros((X.shape[0],n_s)),
                    "c0": np.zeros((X.shape[0],n_s)),
                    "decoder_X": np.zeros((X.shape[0], 1))}
    inputs = Dataset.from_tensor_slices(inputs)
    outputs = {"output": Y_oh}
    outputs = Dataset.from_tensor_slices(outputs)
    return inputs, outputs

# for wrapping into estimator
def read_dataset(csv_path, params, n_s):
    raw_data = load_clean_sentences(csv_path)
    #params = get_tokenizer_helper(raw_data)
    inputs, outputs = create_input_output_tensor(raw_data, params, n_s)
    return Dataset.zip((inputs, outputs))

def train_input_fn(csv_path, params, n_s, batch_size = 128):
    dataset = read_dataset(csv_path, params, n_s)
    dataset = dataset.repeat(count = None).batch(batch_size = batch_size)
    return dataset

def eval_input_fn(csv_path, params, n_s, batch_size = 128):
    dataset = read_dataset(csv_path, params, n_s)
    dataset = dataset.batch(batch_size = batch_size)
    return dataset