""" Captioning Model"""

import os
import keras.layers as layer
import keras.layers.merge as merge
import keras.models as models
import tensorflow.keras.utils as keras_utils


def get_model(vocab_size, max_length):
	# Feature extractor model (extracting features from image features (which were extracted from VGG))
    # VGG features => another feature vector
    # (?, 4096) => (?, 256)
	features_input = layer.Input(shape=(4096,))
	fe1 = layer.Dropout(0.5)(features_input)
	features_output = layer.Dense(256, activation='relu')(fe1)

	# Sequence model
    # Sequence of text => feature vector
    # Sequence of text => Embedding representation => LSTM => feature vector
    # Text => (?, 256)
	text_input = layer.Input(shape=(max_length,))
	se1 = layer.Embedding(vocab_size, 256, mask_zero=True)(text_input)
	se2 = layer.Dropout(0.5)(se1)
	text_output = layer.LSTM(256)(se2)

	# Decoder model
    # Img_feature_vector + Text_feature_vector => (?, vocab_size)
	decoder1 = merge.add([features_output, text_output])
	decoder2 = layer.Dense(256, activation='relu')(decoder1)
	outputs = layer.Dense(vocab_size, activation='softmax')(decoder2)
    
	# Forming the model
	model = models.Model(inputs=[features_input, text_input], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Displaying the model
	print(model.summary())
	# keras_utils.plot_model(model, to_file='model.png', show_shapes=True)
	return model


if __name__ == "__main__":
    get_model(7579, 34)