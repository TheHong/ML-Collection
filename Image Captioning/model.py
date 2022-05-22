""" Captioning Model"""

import os
import pickle
import numpy as np
import keras.layers as layer
import keras.layers.merge as merge
import keras.models as models
import keras.preprocessing.sequence as keras_sequence
import keras.applications.vgg16 as keras_vgg
import keras.preprocessing.image as keras_image
from keras.models import Model
import matplotlib.pyplot as plt
# import tensorflow.keras.utils as keras_utils

import helpers
import config as C


def get_vgg_model():
    original_vgg = keras_vgg.VGG16()
    vgg = Model(inputs=original_vgg.inputs, outputs=original_vgg.layers[-2].output)
    return vgg


def get_head_model(vocab_size, max_length):
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


def run_vgg_on_image(vgg_model, image_path):
    # Load and process image
    image = keras_image.load_img(image_path, target_size=(224, 224))  # Resizing to the preferred input size
    image = keras_image.img_to_array(image)  # Convert to np array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = keras_vgg.preprocess_input(image)

    # Run image through vgg
    features = vgg_model.predict(image, verbose=0)

    return features


def run_on_vgg_features(model, tokenizer, image_vgg_features, max_length):
    """ Builds the description, one word at a time """
    input_text = helpers.START_TOKEN
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]  # Integer encode input sequence
        sequence = keras_sequence.pad_sequences([sequence], maxlen=max_length)

        # Run the model to get a probability distribution
        output = model.predict([image_vgg_features, sequence], verbose=0)

        # Get index of most probably word
        most_probable_word_idx = np.argmax(output)

        # Get word from idx
        word = helpers.word_for_id(most_probable_word_idx, tokenizer)

        # Use the word
        if word is not None:
            input_text += f' {word}' # Add onto the string being built
        else:
            break  # If word could not be found, then stop

        # Stop if the end token is predicted
        if word == helpers.END_TOKEN:
            break

    return input_text


def get_tokenizer():
    with open(C.TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


def run_on_image(nn_head, vgg_backbone, tokenizer, image_path, max_length=34):
	# Run image through VGG to get features
    features = run_vgg_on_image(vgg_backbone, image_path)

    # Generate description using the head network
    raw_description = run_on_vgg_features(nn_head, tokenizer, features, max_length)

    return raw_description


if __name__ == "__main__":
    # CHANGE before you run =======================
    # Specify which model you want to quickly run
    head_model_path = os.path.join(C.MODELS_FOLDER, "image_captioning_model-ep15.h5")
    # Put images on which you want to run the model in the "data" directory
    image_paths = [os.path.join("data", f) for f in os.listdir("data") if f.lower().endswith(".jpg")]
    # ====================================

    # Load the NN components
    print("\nLOADING COMPONENTS =======================\n")
    head = models.load_model(head_model_path)
    backbone = get_vgg_model()
    tokenizer = get_tokenizer()

    for image_path in image_paths:
        # Run the model
        print(f"\nRUNNING THE MODEL on {image_path}===============================\n")
        raw_result = run_on_image(head, backbone, tokenizer, image_path)
        print(f"RAW RESULT '{raw_result}'")
        result = helpers.get_description_from_output(raw_result)
        print(f"RESULT '{result}'")

        # Display results
        plt.imshow(plt.imread(image_path))
        plt.axis('off')
        plt.title(result)
        plt.show()

    # Testing the get_model function
    # get_model(7579, 34)