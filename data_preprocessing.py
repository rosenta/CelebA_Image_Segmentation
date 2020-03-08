import os
import cv2
import time
import random
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from utils.common_utility import CommonUtility
from utils.constant import Constant

# for training
from keras.preprocessing.image import ImageDataGenerator


class CelebADataset:

    def __init__(self):
        self.IMG_HEIGHT = Constant.IMG_HEIGHT
        self.IMG_WIDTH = Constant.IMG_WIDTH
        self.CHANNELS = Constant.CHANNELS
        self.ANNOTATION_DIR = Constant.ANNOTATION_DIR
        self.RAW_IMAGE_DIR = Constant.RAW_IMAGE_DIR
        self.BASE_DIR = Constant.BASE_DIR
        self.anno_folders_dict_list = self.generate_annotated_folder_info()
        CommonUtility.create_folder(self.BASE_DIR)

    def generate_annotated_folder_info(self, ANNOTATION_DIR=None):
        """generates mapping of annotated sub folders and it's respective image 
        
        Keyword Arguments:
            ANNOTATION_DIR {str} -- path of annotation Directory  (default: {None})
        
        Returns:
            [list] -- anno_folders_dict_list
        """
        anno_folders_dict_list = []

        if not ANNOTATION_DIR:
            ANNOTATION_DIR = self.ANNOTATION_DIR

        for folder_num in range(15):          
            anno_folder_dict = {"folder_name": os.path.join(ANNOTATION_DIR, str(folder_num)), "start_idx":folder_num*2000, "end_idx": 2000*(folder_num+1) -1}
            anno_folders_dict_list.append(anno_folder_dict)
        return anno_folders_dict_list

    def find_sub_folder_of_image(self, given_idx):
        """finds location at which annotated mask file of given image file is present 
        
        Arguments:
            given_idx {str} -- file_name (in numbers)
        
        Returns:
            [str] -- folder_name
        """
        for folder_dict in self.anno_folders_dict_list:
            if folder_dict["start_idx"] <= int(given_idx) and int(given_idx) <= folder_dict["end_idx"]:
                return folder_dict["folder_name"]

    def merge_mask(self, file_name, alpha=1, beta=1, gamma=0.0):
        """ merges masks of eye and lips
        
        Arguments:
            file_name {str} -- name of a given file
        
        Keyword Arguments:
            alpha {int} -- parameters related to first base image  (default: {1})
            beta {int} -- parameters related to mask image (default: {1})
            gamma {float} -- parameters for adding weights (default: {0.0})
        
        Returns:
            [ndarray] -- final base_mask or merged mask
        """
        mask_image_data = []
        mask_folder_name = self.find_sub_folder_of_image(file_name)
        base_mask_path = os.path.join(mask_folder_name, "{:05d}_{}.png")

        left_eye_mask = cv2.imread(base_mask_path.format(int(file_name), "l_eye"))
        right_eye_mask = cv2.imread(base_mask_path.format(int(file_name), "r_eye"))
        lower_lip_mask = cv2.imread(base_mask_path.format(int(file_name), "l_lip"))
        upper_lip_mask = cv2.imread(base_mask_path.format(int(file_name), "u_lip"))

        mask_image_data = [left_eye_mask, right_eye_mask, lower_lip_mask, upper_lip_mask]
        # Calculate blended image
        base_mask = np.zeros((self.IMG_WIDTH, self.IMG_HEIGHT, self.CHANNELS), dtype=np.uint8)
        for idx in range(len(mask_image_data)):
            if mask_image_data[idx] is not None:
                base_mask = cv2.addWeighted(mask_image_data[idx], alpha, base_mask, beta, gamma)
        return base_mask
            
    def generate_custom_dataset(self, data_dir, base_path=Constant.BASE_DIR,  dataset_type='train'): 
        """Creates A Custom dataset for lips and eye
        
        Arguments:
            data_dir {str} -- raw Images directory
        
        Keyword Arguments:
            base_path {str} -- base dataset directory (default: {Constant.BASE_DIR})
            dataset_type {str} -- type of dataset data (default: {'train'})
        """
        dataset_mask_path = os.path.join(base_path, dataset_type+"_masks", dataset_type)
        dataset_frame_path = os.path.join(base_path, dataset_type+"_frames", dataset_type)
        CommonUtility.create_folder(dataset_mask_path)
        CommonUtility.create_folder(dataset_frame_path)
        
        mask_base_path = os.path.join(dataset_mask_path, "{}.jpg")
        img_base_path = os.path.join(dataset_frame_path, "{}.jpg")

        start = 0
        step = 12
        
        while start <= len(data_dir):
            processes = []
            for file_path in data_dir[start:start+step]:
                process = multiprocessing.Process(target=self.data_preprocess, args=(file_path, data_dir, mask_base_path, img_base_path) )
                processes.append(process)
            for single_process in processes:
                single_process.start()
            for single_process in processes:
                single_process.join()

            # print("files {} are processing for train dataset".format(data_dir[start:start+step]))
            start = start + step     

    def data_preprocess(self, file_path, data_dir, mask_base_path, img_base_path):
        """Preprocess data in required format
        
        Arguments:
            file_path {str} -- file_path
            data_dir {str} -- data_dir 
            mask_base_path {str} -- mask_base_path
            img_base_path {str} -- img_base_path
        """
        file_name = file_path.split('.jpg')[0]

        # generating single mask image for lips and eyes
        base_mask = self.merge_mask(file_name)            
        _ = cv2.imwrite(mask_base_path.format(file_name), base_mask) 

        # resizing raw image and storing it into respective folder
        # img = cv2.imread(os.path.join(self.RAW_IMAGE_DIR, file_path))
        # base_img = cv2.resize(img, (self.IMG_HEIGHT,self.IMG_WIDTH),0)
        # _ = cv2.imwrite(img_base_path.format(file_name), base_img)

        dest_base_path = os.path.join(img_base_path.format(file_name))
        source_image_path = os.path.join(self.RAW_IMAGE_DIR, file_path)
        CommonUtility.create_copy_of_a_file_to_dest(source_image_path, dest_base_path)

    def mask_folder_creation(self, train_split=0.7):
        """Creates Dataset folders and divides the training and testing data
        
        Keyword Arguments:
            train_split {float} -- train_split (default: {0.7})
        """
        start = time.time()
        base_path = os.path.join(self.BASE_DIR)
        raw_image_list = os.listdir(self.RAW_IMAGE_DIR)
        train_data_count = int(train_split*len(raw_image_list))
        training_data_list = raw_image_list[:train_data_count]
        testing_data_list = raw_image_list[train_data_count:]

        self.generate_custom_dataset(trai[summary]ning_data_list, base_path, dataset_type='train')
        self.generate_custom_dataset(testing_data_list, base_path, dataset_type='test')
        end = time.time()
        time_diff = end - start
        print("took {} seconds or {} minutes to complete custom dataset Generation".format(time_diff, time_diff/60.))

    def data_gen(self, train_frames_dir, train_masks_dir, \
        val_frames_dir, val_masks_dir, \
        rescale=Constant.RESCALE, \
        batch_size=Constant.BATCH_SIZE, \
        shear_range=Constant.SHEAR_RANGE, \
        zoom_range=Constant.ZOOM_RANGE, \
        horizontal_flip=True):
        """Keras ImageDataGenerator with Augumentation    
        Arguments:
            train_frames_dir {str} -- train_frames_dir
            train_masks_dir {str} -- train_masks_dir
            val_frames_dir {str} -- val_frames_dir
            val_masks_dir {str} -- val_masks_dir
        
        Returns:
        train_generator, val_generator {generator} -- train and validation Generator 
        """
        train_datagen = ImageDataGenerator(
            rescale=rescale,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip)
        
        val_datagen = ImageDataGenerator(rescale=rescale)

        train_image_generator = train_datagen.flow_from_directory(train_frames_dir, batch_size = batch_size, class_mode=None, target_size=(Constant.IMG_WIDTH,Constant.IMG_HEIGHT))
        train_mask_generator = train_datagen.flow_from_directory(train_masks_dir, batch_size = batch_size,class_mode=None, target_size=(Constant.IMG_WIDTH,Constant.IMG_HEIGHT), color_mode = "grayscale")
        val_image_generator = val_datagen.flow_from_directory(val_frames_dir, batch_size = batch_size, class_mode=None, target_size=(Constant.IMG_WIDTH,Constant.IMG_HEIGHT))
        val_mask_generator = val_datagen.flow_from_directory(val_masks_dir, batch_size = batch_size, class_mode=None,
        target_size=(Constant.IMG_WIDTH,Constant.IMG_HEIGHT), color_mode = "grayscale")

        train_generator = zip(train_image_generator, train_mask_generator)
        val_generator = zip(val_image_generator, val_mask_generator)
        return train_generator, val_generator

    def custom_data_gen(self, img_folder, mask_folder, batch_size):
        """Custom Generator which which yields raw and mask images
        
        Arguments:
            img_folder {str} -- img_folder
            mask_folder {str} -- mask_folder
            batch_size {str} -- batch_size
        
        Yields:
            img, mask -- img, mask
        """
        count = 0
        train_frames_list = os.listdir(img_folder) #List of training images
        random.shuffle(train_frames_list)
        
        while (True):
            img = np.zeros((batch_size,Constant.IMG_WIDTH, Constant.IMG_HEIGHT, Constant.CHANNELS)).astype('float')
            mask = np.zeros((batch_size, Constant.IMG_WIDTH, Constant.IMG_HEIGHT, 1)).astype('float')

            for idx in range(count, count+batch_size): #initially from 0 to 16, c = 0. 

                train_img = cv2.imread(os.path.join(img_folder, train_frames_list[idx]))/255.
                train_img =  cv2.resize(train_img, (Constant.IMG_WIDTH, Constant.IMG_HEIGHT))# Read an image from folder and resize
                
                img[idx-count] = train_img #add to array - img[0], img[1], and so on.                                                            
                train_mask = cv2.imread(os.path.join(mask_folder, train_frames_list[idx]), cv2.IMREAD_GRAYSCALE)/255.
                train_mask = cv2.resize(train_mask, (Constant.IMG_WIDTH, Constant.IMG_HEIGHT))
                train_mask = train_mask.reshape(Constant.IMG_WIDTH, Constant.IMG_HEIGHT, 1) # Add extra dimension for parity with train_img size [512 * 512 * 3]

                mask[idx-count] = train_mask

            count += batch_size
            if(count+batch_size>=len(os.listdir(img_folder))):
                count=0
                random.shuffle(train_frames_list)
                        # print "randomizing again"
            yield img, mask

    def visualise_data(self, num_images=3):
        """ Function will take num_images as input 
        and plots the dataset mask and images using matplotib subplot

        
        Keyword Arguments:
            num_images {int} -- num_images (default: {3})
        """     
        train_frames_dir = os.path.join(Constant.TRAIN_FRAMES_DIR, 'train')
        train_masks_dir = os.path.join(Constant.TRAIN_MASKS_DIR, 'train')
        raw_images_list = os.listdir(train_frames_dir)

        merge_image_list = []   
        image_list = [] 
        mask_list = []    
        for idx in range(num_images):
            file_path = random.choice(raw_images_list)
            file_name = file_path.split('.jpg')[0]

            img = cv2.imread(os.path.join(train_frames_dir, file_path))
            base_img = cv2.resize(img, (Constant.IMG_WIDTH, Constant.IMG_HEIGHT),0)
            base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
            image_list.append(base_img)

            mask = cv2.imread(os.path.join(train_masks_dir, file_path))
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask_list.append(mask)

            merge_image = cv2.addWeighted(base_img, 0.7, mask, 0.3, 0.0)
            # merge_image = cv2.cvtColor(merge_image, cv2.COLOR_BGR2RGB)
            merge_image_list.append(merge_image)
            

        fig, axs = plt.subplots(nrows=num_images, ncols=3)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.axis('off')

        for plot in range(num_images):
            axs[plot, 0].imshow(image_list[plot], interpolation='nearest')
            axs[plot, 0].set_title('Raw Image')
            axs[plot, 1].imshow(mask_list[plot], interpolation='nearest')
            axs[plot, 1].set_title('Mask Image')           
            axs[plot, 2].imshow(merge_image_list[plot], interpolation='nearest')
            axs[plot, 2].set_title('Segmented Mask')           
        
        plt.show()
            


if __name__ == '__main__':
    dataset_object = CelebADataset()
    dataset_object.mask_folder_creation()
    dataset_object.visualise_data(num_images=2)

    # train_gen, val_gen = dataset_object.data_gen(Constant.TRAIN_FRAMES_DIR, Constant.TRAIN_MASKS_DIR, Constant.VAL_FRAMES_DIR, Constant.VAL_MASKS_DIR)

    # train_gen = dataset_object.custom_data_gen(Constant.TRAIN_FRAMES_DIR+'/train', Constant.TRAIN_MASKS_DIR+'/train', batch_size = 4)
    # val_gen = dataset_object.custom_data_gen(Constant.VAL_FRAMES_DIR+'/test', Constant.VAL_MASKS_DIR+'/test', batch_size = 4)
    