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


def create_sequences_for_one(img_feature, description_list, tokenizer, max_length, vocab_size):
    """
    For a certain image, create sequences based on the image and its corresponding descriptions.
    One sequence consists of the image, input words, and expected output word.
    These will be stored in order in three seperate arrays: X_img, X_txt, y.
    So for an image description that has 10 words, there will be 10 sequences for that image-description pair
    So, for an image that has 5 descriptions, all the sequences corresponding to the 5 image-description pair are stored in the three arrays
    """
    X_img, X_txt, y_word = [], [], []

    for description in description_list:
        seq = tokenizer.texts_to_sequences([description])[0]  # Encode sequence into list of int
        # split one sequence into multiple X,y pairs
        for i in range(len(seq)-1):
            # split into input and output pair
            input = seq[:i+1]  # From first word to current word
            output = seq[i+1]  # The next word

            # Pad input sequence
            in_seq = keras_sequence.pad_sequences([input], maxlen=max_length)[0]
            X_txt.append(in_seq)

            # Encode output sequence
            out_seq = keras_utils.to_categorical([output], num_classes=vocab_size)[0]
            y_word.append(out_seq)

            # Store the image feature corresponding to that description
            X_img.append(img_feature)
            
    return np.array(X_img), np.array(X_txt), np.array(y_word)


def create_sequences_for_all(features, descriptions, tokenizer, max_length, vocab_size):
    X_img, X_txt, y_word = [], [], []
    for name, description_list in tqdm(descriptions.items()):
        img_feature = features[name][0]
        new_sequences = create_sequences_for_one(img_feature, description_list, tokenizer, max_length, vocab_size)
        X_img.extend(new_sequences[0])
        X_txt.extend(new_sequences[1])
        y_word.extend(new_sequences[2])
    return np.array(X_img), np.array(X_txt), np.array(y_word)


def word_for_id(integer, tokenizer):
    """ Finds the word that corresponds to the id """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def get_description_from_output(output):
    # Process the description to remove start and end tokens
    return output.lower().replace(f"{START_TOKEN.lower()} ", "").replace(f" {END_TOKEN.lower()}", "") 