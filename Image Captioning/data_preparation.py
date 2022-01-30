""" This script precomputes the extracted features of the image dataset so that VGG doesn't have to be run everytime the image is used"""

import os
import pickle
from tqdm import tqdm
import string
import keras.applications.vgg16 as keras_vgg
import keras.preprocessing.image as keras_image
from keras.models import Model

import config as C

def prepare_image_data(image_dir):
    """
    First step in the model is extracting features from the image.
    Feature extraction will be done using pre-trained VGG model available on Keras.
    VGG model is a large model.
    To make training faster and use less memory, the features will be computed beforehand.
    This is also ok because we're not fine-tuning the VGG model.
    """

    assert os.path.isdir(image_dir), f"Path '{image_dir}' not found"

    # Remove last layer (which is VGG's classification layer)
    print("Loading VGG16 =======================")
    original_vgg = keras_vgg.VGG16()
    vgg = Model(inputs=original_vgg.inputs, outputs=original_vgg.layers[-2].output)
    print(vgg.summary())

    # Extract features
    features = {}
    print("This will take a while. . . ======================")
    for name in tqdm(os.listdir(image_dir)):
        # Load and process image
        image = keras_image.load_img(os.path.join(image_dir, name), target_size=(224, 224))  # Resizing to the preferred input size
        image = keras_image.img_to_array(image)  # Convert to np array
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = keras_vgg.preprocess_input(image)

        # Get and save features
        feature = vgg.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature

    print('Extracted Features: %d' % len(features)) # Each feature has shape (1, 4096)

    # Save features
    with open(C.FEATURES_FILE_PATH, 'wb') as f:
        pickle.dump(features, f)




def load_descriptions(raw_descriptions_file_path):
    # Get descriptions from file
    with open(raw_descriptions_file_path, 'r') as f:
        doc = f.read()

    # Organize the descriptions by image
    descriptions = {}  # Stores the descriptions of each image (each image has a list of descriptions)
    for line in doc.split('\n'):  # Each line corresponds to a specific image
        # Split line into tokens and only process if there at least 2 tokens on that line
        tokens = line.split()  
        if len(line) < 2:
            continue

        # Extract info from line
        image_filename, description_tokens = tokens[0], tokens[1:]
        image_name = image_filename.split('.')[0]
        image_description = ' '.join(description_tokens)  # Reform the description

        # Save extracted information
        if image_name not in descriptions:
            descriptions[image_name] = []
        descriptions[image_name].append(image_description)
    
    print('Loaded: %d ' % len(descriptions))
    return descriptions


def clean_descriptions(descriptions):
    # Translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    # For each image-descriptions pair
    for description_list in descriptions.values():
        # For each description in the image's description
        for i in range(len(description_list)):
            # Get original string
            description = description_list[i]

            # Process the string
            description = description.split()  # tokenize
            description = [word.lower() for word in description]   # convert to lower case
            description = [w.translate(table) for w in description]  # Remove punctuation
            description = [word for word in description if len(word)>1]  # remove hanging 's' and 'a'
            description = [word for word in description if word.isalpha()]  # remove tokens with numbers in them
            
            # Reform the string
            description_list[i] =  ' '.join(description)


def to_vocabulary(descriptions):
    # Get the full vocabulary based on loaded descriptions
    all_descriptions = set()
    for description_list in descriptions.values():
        for description in description_list:
            all_descriptions.update(description.split())

    return all_descriptions
 

def save_descriptions(descriptions):
    # Save descriptions to file, one description per line
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)

    with open(C.PROCESSED_TEXT_FILE_PATH, 'w') as f:
        f.write('\n'.join(lines))


def prepare_text_data(raw_descriptions_file_path):
    descriptions = load_descriptions(raw_descriptions_file_path)
    clean_descriptions(descriptions)
    save_descriptions(descriptions)

    vocabulary = to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))



if __name__ == "__main__":


    if not os.path.isdir(C.DATA_FOLDER):
        os.makedirs(C.DATA_FOLDER)

    # Processing the images
    if not os.path.isfile(C.FEATURES_FILE_PATH):  # pickle file does not exist
        print("Features file not found. Will generate. =====================")
        prepare_image_data(C.image_dir)
    else:
        print("Features file found. ==========================")

    # Processing the text
    if not os.path.isfile(C.PROCESSED_TEXT_FILE_PATH):  # pickle file does not exist
        print("Processed Descriptions file not found. Will generate.")
        prepare_text_data(C.raw_descriptions_file_path)
    else:
        print("Processed Descriptions file found. ==========================")
