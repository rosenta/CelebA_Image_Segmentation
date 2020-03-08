# for training
import os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from models import unet

from utils.common_utility import CommonUtility
from utils.constant import Constant
import models
from data_preprocessing import CelebADataset



class Runner():
    def train(self, train_model, weights_path=None, epochs=Constant.NUM_EPOCHS, \
        batch_size=Constant.BATCH_SIZE, min_delta=0.01, patience=3, log_path="./training.log", \
            monitor='val_acc',):

        # define generators
        dataset_object = CelebADataset()
        train_gen, val_gen = dataset_object.data_gen(Constant.TRAIN_FRAMES_DIR, Constant.TRAIN_MASKS_DIR, Constant.VAL_FRAMES_DIR, Constant.VAL_MASKS_DIR)

        no_of_train_images = len(os.listdir(os.path.join(Constant.TRAIN_FRAMES_DIR, 'train')))
        no_of_val_images = len(os.listdir(os.path.join(Constant.VAL_FRAMES_DIR, 'test')))

        # checkpoints to be stored
        checkpoint = ModelCheckpoint(weights_path, monitor=monitor, 
                              verbose=1, save_best_only=True, mode='max')
        # define logger
        csv_logger = CSVLogger(log_path, append=True, separator=';')

        # define early stopping
        earlystopping = EarlyStopping(monitor = monitor, verbose = 1,
                                    min_delta = min_delta, patience = patience, mode = 'max')

        # defining set of callbacks
        callbacks_list = [checkpoint, csv_logger, earlystopping]

        results = train_model.fit_generator(train_gen, epochs=epochs, 
                                steps_per_epoch = 100,
                                validation_data=val_gen, 
                                validation_steps=20,
                                callbacks=callbacks_list
                                )
        return results


if __name__ == '__main__':
    
    dataset_object = CelebADataset()
    
    train_gen, val_gen = dataset_object.data_gen(Constant.TRAIN_FRAMES_DIR, Constant.TRAIN_MASKS_DIR, Constant.VAL_FRAMES_DIR, Constant.VAL_MASKS_DIR)
    
    # define model
    weights_path = 'best_model.h5'
    model = models.unet(pretrained_weights=weights_path)
    results = Runner().train(model, weights_path=weights_path, epochs=15, batch_size=Constant.BATCH_SIZE)
    print(results)
 

