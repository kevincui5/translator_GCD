from trainer.util import *
from trainer.global_var import *
import tensorflow as tf
import numpy as np

from nltk.translate.bleu_score import corpus_bleu


n_a = 512
n_s = n_a * 2

params = get_raw_data_tokenizer("english-german.csv")
eng_tokenizer = params.get('eng_tokenizer')
ger_tokenizer = params.get('ger_tokenizer')
vocab_inp_size = params.get('ger_vocab_size')
vocab_tar_size = params.get('eng_vocab_size')
Tx = params.get('ger_length')
Ty = params.get('eng_length')


def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = np.expand_dims(source, axis=0) # add a dimension of 1 as a single sample for prediction of one sentence
		source = [source,np.zeros((source.shape[0],n_s)),np.zeros((source.shape[0],n_s)),np.zeros((source.shape[0],1))]
		translation = predict_sequence(model, tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
         #only display first 10
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append([raw_target.split()])
		predicted.append(translation.split())
    	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

test_sample_size = 112 
#raw_data = load_csv_batch('english-german-train.csv', 0, test_sample_size - 1)
#train_inputs, train_outputs = create_input_output_tensor(raw_data, params, n_s)
#evaluate train set
train_inputs, _, train_raw = load_data_helper('train_data/english-german-train.csv', 0, test_sample_size, n_s, params)
trainX = train_inputs[0]

#model = load_model('translation_model3.h5')
#model = tf.saved_model.load('trained_model') #not a Keras object (i.e. doesn't have .fit, .predict, etc. methods)
model =  tf.keras.models.load_model('trained_model', custom_objects={'tf': tf}) # interestingly I did not have to specify 'tf' as a custom object when the load function was called in the same folder as the corresponding save function. Once I moved my load-call to another folder I had do specify it. This sure looks like a bug
#model.load_weights('translation_model3.h5')
#evaluate_model(model, eng_tokenizer, train_inputs.get("X")[:test_sample_size], train_outputs.get("output")[:test_sample_size])
evaluate_model(model, eng_tokenizer, trainX[:test_sample_size], train_raw[:test_sample_size])
#evaluate test set
#test_raw  = load_clean_sentences('english-german-test.csv')
test_inputs, _, test_raw = load_data_helper('english-german-test.csv', 0, test_sample_size, n_s, params)
testX = test_inputs[0]
evaluate_model(model, eng_tokenizer, testX[:test_sample_size], test_raw[:test_sample_size])
#test_inputs, test_outputs = create_input_output_tensor(test_raw, params, n_s) #2nd para, total line number in file
#testX = test_inputs[0]

#evaluate_model(model, eng_tokenizer, test_inputs.get("X"), test_outputs.get("output"))
#evaluate_model(model, eng_tokenizer, testX, test_raw)
