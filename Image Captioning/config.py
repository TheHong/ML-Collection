import os


# TODO: Change the below accordingly ==========================================
# Absolute paths
image_dir = ""  # Path to "Flickr8k_Dataset\\Flicker8k_Dataset"
raw_descriptions_file_path = ""  # Path to "Flickr8k_text\\Flickr8k.token.txt"
trs_names_file_path = ""  # Path to "Flickr8k_text\\Flickr_8k.trainImages.txt"
val_names_file_path = ""  # Path to "Flickr8k_text\\Flickr_8k.devImages.txt"
tes_names_file_path = ""  # Path to "Flickr8k_text\\Flickr_8k.testImages.txt"
# =============================================================================

DATA_FOLDER = "data"
MODELS_FOLDER = "models"
FEATURES_FILE_PATH = os.path.join(DATA_FOLDER, "image_captioning_vgg_features.pkl")
PROCESSED_TEXT_FILE_PATH = os.path.join(DATA_FOLDER, "image_captioning_descriptions.txt")
TOKENIZER_PATH = os.path.join(MODELS_FOLDER, "image_captioning_tokenizer.pkl")
