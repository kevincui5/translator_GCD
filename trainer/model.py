# -*- coding: utf-8 -*-
from trainer.util import *
from trainer.global_var import *
from tensorflow.keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape
import numpy as np
#from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from nltk.translate.bleu_score import corpus_bleu
import json  #for json callback
from datetime import datetime
import tensorflow as tf
#TRAIN BLEU: 0.94, 0.915, 0.89, 0.78
#TEST BLEU: 0.65, 0.55, 0.50, 0.365
# use teachers forcing in decoder.  inference and training models are the same



# define model  
def build_model(Tx, Ty, vocab_inp_size, vocab_tar_size, n_s, n_a, embedding_dim):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    embedding_dim -- embedding layer output size
    n_s -- hidden state size of the post-attention LSTM
    n_a -- hidden state size of the pre-attention Bi-LSTM
    vocab_inp_size -- size of the python dictionary "vocab_inp_size"
    vocab_tar_size -- size of the python dictionary "vocab_tar_size"

    Returns:
    inference_model -- Keras inference model instance
    """
    
    # Define the inputs of your model with a shape (Tx, vocab_inp_size)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X0 = Input(shape=(Tx, ),name='X')
    # (m,Tx)
    encoder_embedding = Embedding(vocab_inp_size, embedding_dim, input_length=Tx)
    X = encoder_embedding(X0)
    # (m,Tx,embedding_dim)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    decoder_X0 = Input(shape=(1,), name='decoder_X')
    decoder_X = decoder_X0
    #shape=(m, 1)
    #shape is not (m, Ty) because we manually iterate Ty timesteps
    
    #Define encoder as Bi-LSTM
#multi layer gives worse result, 2% lower bleu score    
#    enc_layers = 1
#    for i in range(enc_layers):
#        X = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    encoder_layer = Bidirectional(LSTM(n_a, return_sequences=True, return_state = True))
    encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_layer(X)
    encoder_hidden = Concatenate(axis=-1)([forward_h, backward_h])
    encoder_cell = Concatenate(axis=-1)([forward_c, backward_c])
    
    s = encoder_hidden
    c = encoder_cell
    #beginning of attention code
    repeator = RepeatVector(Tx)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation = "tanh")
    densor2 = Dense(1, activation = "relu")
    #activator = Activation(softmax, name='attention_weights') # customed softmax
    activator = Activation('softmax', name='attention_weights')
    dotor = Dot(axes = 1)
    def one_step_attention(a, s_prev):
        """
        Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        
        Arguments:
        a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        
        Returns:
        context -- context vector, input of the next (post-attetion) LSTM cell
        """
        
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
        concat = concatenator([a, s_prev])
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
        context = dotor([alphas, a])
        
        return context
    #end of attention code
    #decoder layers initialization is put here instead of global_var.py because they are called inside the for loop for Ty times in function model and
    #share the weights would make training faster
    
    decoder_cell = LSTM(n_s, return_state = True)
    output_layer = Dense(vocab_tar_size, activation='softmax')
    #reshape_layer = Reshape((1,))
    enc_concat = Concatenate(axis=-1)
    decoder_embedding = Embedding(vocab_tar_size, embedding_dim, input_length=1)
    #using pre-trained weights in word embedding layer
    #embedding_matrix = get_embedding_matrix("glove.twitter.27B.100d.txt", 
    #                                        eng_vocab_size, eng_tokenizer)
    #decoder_embedding = Embedding(vocab_tar_size, embedding_dim, 
    #                              weights=[embedding_matrix], input_length=1,
    #
    for t in range(Ty):
        # one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(encoder_output, encoder_hidden)
        #(m, 1, n_s)
        decoder_X = decoder_embedding(decoder_X)
        #(m,1) - (m,1,embedding_dim)
        decoder_inputs = enc_concat([decoder_X, context])
        #shape--(m, 1, n_s+embedding_dim)
        # Apply the post-attention LSTM cell to the "context" vector.
        #initial_state = [hidden state, cell state]
        decoder_X, s, c = decoder_cell(decoder_inputs, initial_state = [s, c])
        #decoder_X.shape--(m,n_s)
        # Step 2.C: Apply Dense layer to the hidden state output of the decoder LSTM
        decoder_X = output_layer(decoder_X)
        #(m,vocab_tar_size)
        out = decoder_X
        #trick to add a dimension of 1 to tensor
        decoder_X = RepeatVector(1)(decoder_X)
        decoder_X = Lambda(lambda x: tf.keras.backend.argmax(x))(decoder_X) #sampling
        #shape--(m,1) so that it can fit embedding layer
        outputs.append(out) #(Ty,m,vocab_tar_size)
    vocab_tar_size = outputs[0].shape[1]
    #None is the batch size (Ty,m,vocab_tar_size)->(m,Ty,vocab_tar_size)
    outputs = Reshape((Ty, vocab_tar_size), name="output")(Concatenate()(outputs)) #
#    dec_layers = 3
#    for i in range(dec_layers):
        #print(outputs) 
#        decoder_outputs = LSTM(n_s, return_sequences = True)(decoder_outputs)
    model = Model([X0, s0, c0, decoder_X0], outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# summarize defined model
#print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
#teacher forcing: np.append(np.zeros((trainY.shape[0],1)),trainY[:,:-1],1)

    
def train_and_evaluate(args):
    n_a = args["n_a"]
    n_s = n_a * 2
    #embedding_dim = 256
    embedding_dim = args["embedding_dim"] #from the glove embedding text, dimension 100
    train_examples = args["train_examples"] #each is 1000
    #test_m = 9645
    batch_size = args["batch_size"]
    steps_per_epoch = train_examples // batch_size
    #train_m = 38 #for testing
    #test_m = 2 #for testing
    #steps_per_epoch = 256 #for testing
    #batch_size = 156
    num_epochs = args["num_epochs"]
    #num_epochs = 1 #for testing
       
    params = get_raw_data_tokenizer(args["complete_data_path"])
    eng_tokenizer = params.get('eng_tokenizer')
    ger_tokenizer = params.get('ger_tokenizer')
    vocab_inp_size = params.get('ger_vocab_size')
    vocab_tar_size = params.get('eng_vocab_size')
    Tx = params.get('ger_length')
    Ty = params.get('eng_length')
    
    model = build_model(Tx, Ty, vocab_inp_size, vocab_tar_size, n_s, n_a, embedding_dim)
    ### Generator objects for train and validation
    #training_batch_generator = batch_generator('english-german-train.csv', batch_size, steps_per_epoch)
    #testing_batch_generator = batch_generator('english-german-test.csv', data_batch_size,steps_per_epoch)
    
    dataset = train_input_fn(args["train_data_path"], params, n_s, batch_size)
    
    model_weights = 'models/weights'
    #checkpoint = ModelCheckpoint(modelfilename, monitor='loss', verbose=0, save_best_only=True, mode='min')
    save_weights = tf.keras.callbacks.ModelCheckpoint(filepath=model_weights,monitor='loss',
                                                 save_weights_only=True, save_best_only=True,
                                                 verbose=0)
    early_stopping = EarlyStopping(monitor='loss', patience=2)
    '''
    json_log = open('debug_train_log.json', mode='wt', buffering=1)
    json_logging_callback = tf.keras.callbacks.LambdaCallback(
        on_batch_begin=lambda batch, logs: json_log.write(
            json.dumps({'time': datetime.now().strftime("%H:%M:%S%f"), 'train batch# starts': batch}) + '\n'),
        on_batch_end=lambda batch, logs: json_log.write(
            json.dumps({'time': datetime.now().strftime("%H:%M:%S%f"), 'train batch# end': batch}) + '\n'),
        on_train_end=lambda logs: json_log.close()
    )'''
    #in callback, use metric "val_loss" only when validation_data is provided, or there will be warning and the callback won't work
    #model.fit(train_inputs, train_outputs, epochs=20, batch_size=batch_size, validation_data=(test_inputs, test_outputs), callbacks=[early_stopping, checkpoint])
    model.fit(dataset, epochs=num_epochs, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping])
    #model.fit(training_batch_generator, epochs=20, validation_data=(test_inputs, test_outputs), callbacks=[early_stopping, checkpoint])
    #model.fit(x = training_batch_generator,epochs=num_epochs,steps_per_epoch=steps_per_epoch,verbose=1, callbacks=[json_logging_callback])
    #model.save('translation_model3.h5', save_format="h5")
    tf.saved_model.save(obj=model, export_dir=args["output_dir"])
    #model=tf.saved_model.load('models')
    
#train_and_evaluate()



