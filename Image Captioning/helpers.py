import pickle
import numpy as np
from tqdm import tqdm
import keras.preprocessing.text as keras_text
import keras.preprocessing.sequence as keras_sequence
import tensorflow.keras.utils as keras_utils

import config as C


START_TOKEN = "startToken"
END_TOKEN = "endToken"

def load_image_names(image_names_file):
    """ Loads dataset based on the image names given in the image_names_file """
    with open(image_names_file, 'r') as f:
        filenames = f.read().split("\n")
    
    img_names = []
    for filename in filenames:
        if len(filename) != 0:
            name = filename.split('.')[0]
            img_names.append(name)
    
    return set(img_names)

def load_descriptions(description_file, img_names):
    with open(description_file, 'r') as f:
        lines = f.read().split("\n")

    descriptions = {}
    for name_with_tokens_str in lines:
        name_with_tokens = name_with_tokens_str.split()
        name, description_tokens = name_with_tokens[0], name_with_tokens[1:]
        if name in img_names:
            if name not in descriptions:
                descriptions[name] = []
            description = f"{START_TOKEN} {' '.join(description_tokens)} {END_TOKEN}"
            descriptions[name].append(description)
    return descriptions

def load_features(features_file, img_names):
    with open(features_file, 'rb') as f:
        all_features = pickle.load(f)

    features = {k: all_features[k] for k in img_names}
    return features

def get_vocab_size(tokenizer):
    return len(tokenizer.word_index) + 1

def get_descriptions_as_list(descriptions):
    all_description_list = []
    for description_list in descriptions.values():
        all_description_list.extend([d for d in description_list])
    return all_description_list

def get_max_length(descriptions):
	lines = get_descriptions_as_list(descriptions)
	return max(len(d.split()) for d in lines)

def create_tokenizer(descriptions):
    # Combine all the descriptions in the dict into a list
    all_description_list = get_descriptions_as_list(descriptions)
    
    # Fit tokenizer based on the descriptions
    tokenizer = keras_text.Tokenizer()
    tokenizer.fit_on_texts(all_description_list)

    return tokenizer


# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, features, vocab_size):
    """
    For each image, create a sequence consisting of the image, input words, and expected output word.
    These will be sorted in three seperate arrays: X_img, X_txt, y
    """

    X_img, X_txt, y = [], [], []

    for name, description_list in tqdm(descriptions.items()):
        for description in description_list:
            seq = tokenizer.texts_to_sequences([description])[0]  # Encode sequence into list of int
            # split one sequence into multiple X,y pairs
            for i in range(len(seq)-1):
                # split into input and output pair
                input = seq[:i+1]  # From first word to current word
                output = seq[i+1]  # The next word

                # Pad input sequence
                in_seq = keras_sequence.pad_sequences([input], maxlen=max_length)[0]

                # Encode output sequence
                out_seq = keras_utils.to_categorical([output], num_classes=vocab_size)[0]

                # Store
                X_img.append(features[name][0])
                X_txt.append(in_seq)
                y.append(out_seq)

    # return np.array(X_img), np.array(X_txt), np.array(y)
    return X_img, X_txt, y


def load_ds(ds_names_file_path, verbose=False):
    """ Uses file paths from config.py """
    def _print(str):
        if verbose: 
            print(str)

    # Load the names in the dataset
    ds_names = load_image_names(ds_names_file_path)
    _print(f'Dataset: {len(ds_names)}')

    # Load descriptions (dict[img_name: list[description_str]])
    descriptions = load_descriptions(C.PROCESSED_TEXT_FILE_PATH, ds_names)
    max_length = get_max_length(descriptions)
    _print(f'Descriptions: train={len(descriptions)}')

    # Load features (dict[img_name: feature_vector])
    features = load_features(C.FEATURES_FILE_PATH, ds_names)
    _print(f'Image Features: train={len(features)}')

    # Create tokenizer
    tokenizer = create_tokenizer(descriptions)
    vocab_size = get_vocab_size(tokenizer)
    _print(f"Vocabulary Size = {vocab_size}")

    # Create sequences to be used to fit the model
    X_img, X_txt, y = create_sequences(
        tokenizer, 
        max_length, 
        descriptions, 
        features, 
        vocab_size
    )

    return X_img, X_txt, y, vocab_size, max_length

if __name__ == "__main__":
    # Load the dataset from the training dataset (6K points)
    load_ds(C.trs_names_file_path, verbose=True)
