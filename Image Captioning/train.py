import os
import keras.callbacks as keras_callbacks

from model import get_model
import helpers
import config as C

if __name__ == "__main__":
    trs_X_img, trs_X_txt, trs_y, vocab_size, max_length = helpers.load_ds(C.trs_names_file_path)
    val_X_img, val_X_txt, val_y, _, _ = helpers.load_ds(C.val_names_file_path)

    # Create model
    model = get_model(vocab_size, max_length)

    # Define checkpoint callback
    filepath = os.path.join(C.MODELS_FOLDER, 'captioning_model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    checkpoint = keras_callbacks.ModelCheckpoint(
        filepath, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, 
        mode='min'
    )

    # Train
    model.fit(
        [trs_X_img, trs_X_txt], 
        trs_y, 
        epochs=20, 
        verbose=2, 
        callbacks=[checkpoint], 
        validation_data=([val_X_img, val_X_txt], val_y)
    )