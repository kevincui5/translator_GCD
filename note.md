a simple version of seq-seq model:

# Seq2Seq model

latent_dim = 256

enc_input = Input(shape=(None, ), name='enc_input')
enc_emb = Embedding(input_dim=self.p_vocab_size, output_dim=128, input_length=self.maxlen, name='enc_emb')
enc_lstm = Bidirectional(CuDNNLSTM(units=latent_dim, return_state=True, name='enc_lstm'))
enc_out, forward_h, forward_c, back_h, back_c = enc_lstm(enc_emb(enc_input))

enc_states = [Concatenate()([forward_h, back_h]), Concatenate()([forward_c, back_c])]

dec_input = Input(shape=(None, self.c_vocab_size), name='dec_input')

dec_lstm = CuDNNLSTM(units=latent_dim*2, return_sequences=True, return_state=True, name='dec_lstm')
dec_out, _, _ = dec_lstm(dec_input, initial_state=enc_states)
dec_dense = Dense(units=self.c_vocab_size, activation='softmax', name='dec_dense')
dec_target = dec_dense(dec_out)

model = Model([enc_input, dec_input], dec_target)
model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
model.load_weights(inf_weight_filepath)

# inference
enc_model = Model(enc_input, enc_states)

dec_state_in = [Input(shape=(latent_dim*2,)), Input(shape=(latent_dim*2,))]
inf_dec_out, h, c = dec_lstm(dec_input, initial_state=dec_state_in)
dec_states = [h, c]
dec_out = dec_dense(inf_dec_out)

dec_model = Model([dec_input] + dec_state_in, [dec_out] + dec_states)


for testing on local on small dataset
gcloud ai-platform local train \
        --module-name=trainer.task \
        --package-path="./trainer" \
        -- \
        --train_data_path="./train_data/english-german-train.csv" \
        --n_a=512 \
        --batch_size=156 \
        --embedding_dim=100 \
        --num_epochs=1 \
        --train_examples=1 \
        --job-dir="./job" \
        --output_dir="./trained_model" \
        --complete_data_path="./english-german.csv"
        

export BUCKET_NAME="translator-gcd"
export REGION="us-central1"
export JOB_NAME="translator_traning2"
export JOB_DIR="gs://$BUCKET_NAME/job-dir"
cd OneDrive/courses/tf_ws/translator_GCD/
ls -pR
gsutil -m cp *.csv gs://$BUCKET_NAME/data/
gsutil -m cp -r train_data gs://$BUCKET_NAME/



gcloud ai-platform jobs submit training $JOB_NAME \
      --package-path=trainer/ \
      --module-name=trainer.task \
      --region=$REGION \
      --python-version=3.7 \
      --runtime-version=2.4 \
      --job-dir=$JOB_DIR \
      -- \
    --train_data_path gs://$BUCKET_NAME/train_data/english-german-train.csv \
    --n_a 512 \
    --batch_size 156 \
    --embedding_dim 100 \
    --num_epochs 1 \
    --train_examples 1 \
    --output_dir gs://$BUCKET_NAME/trained_model \
    --complete_data_path gs://$BUCKET_NAME/data/english-german.csv