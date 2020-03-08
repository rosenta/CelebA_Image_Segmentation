# for training
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
    def train(self, train_model, epochs, batch_size):

        # define generators
        dataset_object = CelebADataset()
        train_gen, val_gen = dataset_object.data_gen(Constant.TRAIN_FRAMES_DIR, Constant.TRAIN_MASKS_DIR, Constant.VAL_FRAMES_DIR, Constant.VAL_MASKS_DIR)

        no_of_train_images = len(os.path.join(Constant.TRAIN_FRAMES_DIR, 'train'))
        no_of_val_images = len(os.path.join(Constant.VAL_FRAMES_DIR, 'test'))


        # checkpoints to be stored
        weights_path = 'checkpoints'

        checkpoint = ModelCheckpoint(weights_path, monitor=['accuracy','loss'], 
                                    verbose=1, save_best_only=True, mode='max')

        # define logger
        csv_logger = CSVLogger('./log.out', append=True, separator=';')

        # define early stopping
        earlystopping = EarlyStopping(monitor = 'accuracy', verbose = 1,
                                    min_delta = 0.01, patience = 3, mode = 'max')

        # defining set of callbacks
        callbacks_list = [checkpoint, csv_logger, earlystopping]

        results = train_model.fit_generator(train_gen, epochs=epochs, 
                                steps_per_epoch = no_of_train_images//Constant.BATCH_SIZE,
                                validation_data=val_gen, 
                                validation_steps=(no_of_val_images//Constant.BATCH_SIZE), 
                                )

        return results


if __name__ == '__main__':
    
    dataset_object = CelebADataset()
    
    train_gen, val_gen = dataset_object.data_gen(Constant.TRAIN_FRAMES_DIR, Constant.TRAIN_MASKS_DIR, Constant.VAL_FRAMES_DIR, Constant.VAL_MASKS_DIR)
    
    # define model
    model = models.unet()
    results = Runner().train(model, epochs=Constant.NUM_EPOCHS, batch_size=Constant.BATCH_SIZE)
 

