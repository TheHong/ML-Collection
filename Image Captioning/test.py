import os
from tqdm import tqdm
import keras.models as keras_models
from nltk.translate.bleu_score import corpus_bleu

import model as M
import data_loading
import config as C

# evaluate the skill of the model
def evaluate_model(model, features, descriptions, tokenizer, max_length):
	actual, predicted = [], []
	# step over the whole set
	for name, description_list in tqdm(descriptions.items()):
		# generate description
		output = M.run_on_vgg_features(model, tokenizer, features[name], max_length)
		# store actual and predicted
		references = [d.split() for d in description_list]
		actual.append(references)
		predicted.append(output.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


if __name__ == "__main__":
    model_path = os.path.join(C.MODELS_FOLDER, "captioning_model-ep3.h5")

    # Get data
    features, descriptions, _, _, vocab_size = data_loading.get_ds_info(C.tes_names_file_path, verbose=True, save_tokenizer=False)

    # Set info from training
    tokenizer = M.get_tokenizer()
    max_length = 34

    # Get model
    model = keras_models.load_model(model_path)

    # Evaluate model
    evaluate_model(model, features, descriptions, tokenizer, max_length)
