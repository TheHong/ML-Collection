import pickle

import helpers
import config as C

def get_ds_info(ds_names_file_path, verbose=False, save_tokenizer=True):
    """ Uses file paths from config.py """
    def _print(str):
        if verbose: 
            print(str)

    # Load the names in the dataset
    ds_names = helpers.load_image_names(ds_names_file_path)
    _print(f'Dataset: {len(ds_names)}')

    # Load descriptions (dict[img_name: list[description_str]])
    descriptions = helpers.load_descriptions(C.PROCESSED_TEXT_FILE_PATH, ds_names)
    max_length = helpers.get_max_length(descriptions)
    _print(f'Descriptions: train={len(descriptions)}')

    # Load features (dict[img_name: feature_vector])
    features = helpers.load_features(C.FEATURES_FILE_PATH, ds_names)
    _print(f'Image Features: train={len(features)}')

    # Create tokenizer
    tokenizer = helpers.create_tokenizer(descriptions)
    if save_tokenizer:
        with open(C.TOKENIZER_PATH, 'wb') as f:
            pickle.dump(tokenizer, f)
    vocab_size = helpers.get_vocab_size(tokenizer)
    _print(f"Vocabulary Size = {vocab_size}")

    return features, descriptions, tokenizer, max_length, vocab_size


def load_ds(ds_names_file_path, verbose=False):
    features, descriptions, tokenizer, max_length, vocab_size = get_ds_info(ds_names_file_path, verbose)

    # Create sequences to be used to fit the model
    X_img, X_txt, y = helpers.create_sequences_for_all(
        features, 
        descriptions, 
        tokenizer, 
        max_length, 
        vocab_size
    )

    return X_img, X_txt, y, vocab_size, max_length



def load_data_generator(features, descriptions, tokenizer, max_length, vocab_size):
    # Keep lopping over the images
    while 1:
        # Go through the images in the descriptions dict
        for name, description_list in descriptions.items():
            img_feature = features[name][0]
            X_img, X_txt, y_word = helpers.create_sequences_for_one(
                img_feature, 
                description_list,
                tokenizer, 
                max_length, 
                vocab_size
            )
            yield [X_img, X_txt], y_word


if __name__ == "__main__":
    # # Load the dataset from the training dataset (6K points)
    # load_ds(C.trs_names_file_path, verbose=True)  # Takes a very long time

    # Using a generator for progressive loading
    features, descriptions, tokenizer, max_length, vocab_size = get_ds_info(C.trs_names_file_path, verbose=True)
    generator = load_data_generator(features, descriptions, tokenizer, max_length, vocab_size)
    inputs, outputs = next(generator)
    print(inputs[0].shape)
    print(inputs[1].shape)
    print(outputs.shape)
