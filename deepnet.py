import numpy as np
from keras.layers import LSTM, Dense, Input
from keras.models import Model
#hyperparameters
epochs = 100
latent_dim = 512
batch_size = 250
num_samples = 1000
#get data and vectorize it
data = open('spa.txt', 'r').read().splitlines()[:num_samples]
english = [i.split('\t')[0].lower() for i in data]
spanish = ['\t' + i.split('\t')[1].lower() + '\n' for i in data]
english_vocab = sorted({letter for word in english for letter in word})
spanish_vocab = sorted({letter for word in spanish for letter in word})
num_english_features = len(english_vocab)
num_spanish_features = len(spanish_vocab)
max_encoder_size = max([len(i) for i in english])
max_decoder_size = max([len(i) for i in spanish])
eng_c2i = {c:i for i,c in enumerate(english_vocab)}
spa_c2i = {c:i for i,c in enumerate(spanish_vocab)}
eng_i2c = {i:c for c,i in eng_c2i.items()}
spa_i2c = {i:c for c,i in spa_c2i.items()}

encoder_input_data = np.zeros(shape = (num_samples, max_encoder_size, num_english_features))
decoder_input_data = np.zeros(shape = (num_samples, max_decoder_size, num_spanish_features))
decoder_target_data = np.zeros(shape = (num_samples, max_decoder_size, num_spanish_features))

for index in range(num_samples):
    for i, c in enumerate(english[index]):
        encoder_input_data[index, i, eng_c2i[c]] = 1
    for i, c in enumerate(spanish[index]):
        decoder_input_data[index, i, spa_c2i[c]] = 1
        if i > 0:
            decoder_target_data[index, i-1, spa_c2i[c]] = 1

encoder_input = Input(shape = (None, num_english_features))
encoder_lstm = LSTM(latent_dim, return_state = True)
_, encoder_state_h, encoder_state_c = encoder_lstm(encoder_input)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_input = Input(shape = (None, num_spanish_features))
decoder_lstm = LSTM(latent_dim, return_sequences = True, return_state = True)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state = encoder_states)
decoder_dense = Dense(num_spanish_features, activation = 'softmax')
dense_output = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], dense_output)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs = epochs, batch_size = batch_size, validation_split = .1)

encoder_model = Model(encoder_input, encoder_states)

decoder_input_h = Input(shape = (latent_dim,))
decoder_input_c = Input(shape = (latent_dim,))
decoder_state_inputs = [decoder_input_h, decoder_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_input, initial_state = decoder_state_inputs)
decoder_state_outputs = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(decoder_state_inputs + [decoder_input], [decoder_outputs] + decoder_state_outputs)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    decoder_data = np.zeros((1, 1, num_spanish_features))
    decoder_data[0, 0, spa_c2i['\t']] = 1

    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict(states_value+[decoder_data])
        char_id = np.argmax(output_tokens[0, 0, :])
        char = spa_i2c[char_id]
        decoded_sentence += char
        if char == '\n' or len(decoded_sentence) > max_decoder_size:
            break

        decoder_data = np.zeros((1, 1, num_spanish_features))
        decoder_data[0, 0, char_id] = 1

        states_value = [h, c]

    return decoded_sentence

for seq_index in range(1):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', english[seq_index])
    print('Decoded sentence:', decoded_sentence)
