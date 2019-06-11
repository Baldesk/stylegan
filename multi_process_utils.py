#############################################################
# Filename: multi_process_utils.py
# Description: Defines helpful functions for multiprocessing
# Author: Kyle Baldes
# Date: 1/31/2019
#
#############################################################
import multiprocessing
import numpy as np
import cv2
import os
#import face_data_handler
#import parse_json_configs
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class OneToOne:
    def __init__(self):
        self.data = {}
    def add(self, val1, val2):
        self.data[val1] = val2
        self.data[val2] = val1
    def remove(self, val1):
        self.data.pop(self.data.pop(val1))
    def get(self, val1):
        return self.data[val1]

# ids_labels = OneToOne()
# ids_labels.add()

# def import_img_list_dirs(dir_path):
#     return multi_thread(import_img_dir(dir_path), )

###################################################################################################
# Function: time_func
# Parameters:
# Returns:
# Example:
#
###################################################################################################
def time_func(func_ptr, args_list, mesg):
    with func_ptr as func:
        t_start = time.time()
        ret = func(args_list)
        t_end = time.time()
        print(mesg + str(t_end-t_start))
    return ret
###################################################################################################
# Function: import_img_dir
# Parameters:
# Returns:
# Example:
#
###################################################################################################
def import_img_dir(dir_path):
    file_list = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path)]
    return multi_thread(get_img, file_list, multiprocessing.cpu_count())

# Returns a list
def list_dir_paths(parent_dir):
    return [os.path.join(parent_dir, file_name) for file_name in os.listdir(parent_dir)]



# Saves an image to the path specified
# Note: input is a tuple (filename, img)
def save_img(my_iter):
    filename, img = my_iter
    #print(filename)
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))







# Given a list of images and corresponding labels
# Returns a dictionary where:
#                   keys: label
#                   content: idx list of images belonging to that label
# def get_list_by_label(label_list, num_classes):
#     list_by_label = {}





# Saves a list of images to the appropriate subdirectory for each label in parent_dir
def save_imgs_to_dir(parent_dir, img_list, labels_list):
    subdir_names = np.unique(labels_list) # Get list of labels for subdirectory names
    subdir_paths = [os.path.join(parent_dir, sub_dir) for sub_dir in subdir_names]
    for s_dir in subdir_paths:
        if not os.path.isdir(s_dir):
            os.mkdir(s_dir) # Make sure there is a directory corresponding to the labels
    idx_by_label = split_data_bylabel(labels_list, len(subdir_names))
    # Get list of images for each label
    # For each label
        # Save list of images
    for s_label in subdir_paths:
        p_dir, subdir_n = os.path.split(s_label)
        save_imgs_dir(s_label, img_list[idx_by_label[subdir_n]])

###################################################################################################
# Function: get_img
# Parameters: Path iterable
# Returns: Image from path specified as a numpy array in RGB format
# Example:
#   utils.seq_loop(get_img, path_list_iter, 5)
###################################################################################################
def get_img(my_it):
    img = np.uint8(np.array(cv2.cvtColor(cv2.imread(my_it), cv2.COLOR_BGR2RGB)))
    #print(np.shape(img))
    return img
    #return np.array(cv2.cvtColor(cv2.imread(my_it), cv2.COLOR_BGR2RGB))

###################################################################################################
# Function: seq_loop
# Description:
# Parameters:
#       1. func_ptr - Function pointer to the function that should be called
#       2. my_iter - Iterable to loop through when mapping the worker pool
#       3. num_processes - Number of processes that should be assigned to the worker pool
# Returns: The return values of the Worker pool formated into a numpy array
#          NOTE: The return value corresponds directly to the order of my_iter.
###################################################################################################
def seq_loop(func_ptr, my_iter, num_processes):
    p = multiprocessing.Pool(num_processes)
    ret_arr = p.map(func_ptr, my_iter)
    p.close()
    p.join()
    return np.array(ret_arr)



def get_img_256(my_it):
    img = np.uint8(cv2.resize(np.array(cv2.cvtColor(cv2.imread(my_it), cv2.COLOR_BGR2RGB)),(256,256)))
    #print(np.shape(img))
    return img




def random_flip(img_in):
    if np.random.randint(0,1):
        img_in = np.fliplr(img_in)
    return img_in

##############################################################################################
# Function: multi_thread
# Description: Executes a given function using THREAD based parallelism
# Parameters:
#           * func -> function to be called
#           * args -> arguments to be passed into func
#                  ? - must be iterable
#           * num_workers -> maximum number of workers to use
# Return: List of returned values from func
# Notes: BEST FOR IO INTENSIVE WORKLOADS
# Reference: https://github.com/bfortuner/ml-study/blob/master/multitasking_python.ipynb
##############################################################################################
def multi_thread(func, args, num_workers):
    with ThreadPoolExecutor(num_workers) as execute:
        res = execute.map(func, args)
    return list(res)

##############################################################################################
# Function: multi_process
# Description: Executes a given function using PROCESS based parallelism
# Parameters:
#           * func -> function to be called
#           * args -> arguments to be passed into func
#                  ? - must be iterable
#           * num_workers -> maximum number of workers to use
# Return: List of returned values from func
# Notes: BEST FOR CPU INTENSIVE WORKLOADS
# Reference: https://github.com/bfortuner/ml-study/blob/master/multitasking_python.ipynb
##############################################################################################
def multi_process(func, args, num_workers):
    with ProcessPoolExecutor(num_workers) as execute:
        res = execute.map(func, args)
    return list(res)



def single_process_fork(start_func, vals):
    p = multiprocessing.Process(target=start_func, args=(vals))
    p.start()
    p.join()





def rand_img_augment():
    # Get Training config
    #train_config = parse_json_configs.TrainConfig()

    data_handler = face_data_handler.FaceDataHandler("/home/kyle/Downloads/vgg2_train/",
                                                     "/home/kyle/Downloads/vgg2_test/",
                                                     "/home/kyle/Downloads/identity_meta.csv",
                                                     os.getcwd()+"/dataset_files/improved_50/",
                                                     50,
                                                     False)
    r1 = np.random.randint(0, len(data_handler.tr_paths)-1)
    r2 = np.random.randint(0, len(data_handler.tr_paths)-1)
    img1 = data_handler.get_aligned_image_train(data_handler.tr_paths[r1], data_handler.tr_boxes[r1])
    img2 = data_handler.get_aligned_image_train(data_handler.tr_paths[r2], data_handler.tr_boxes[r2])
    cv2.imwrite('./temp/img1.png', img1)
    cv2.imwrite('./temp/img2.png', img2)
    #maybe_flipped = random_flip(img1)
    #cv2.imshow('Maybe Flipped', maybe_flipped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #if cv2.waitKey(25) & 0xFF == ord('q'):



# ONLY USED FOR TESTING FUNCTIONS
if __name__ == "__main__":
    # Test seq_loop
    rand_img_augment()




