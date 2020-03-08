# -*- coding: utf-8 -*-
__author__ = "Roshan Siyaram Chauhan"
__copyright__ = "Copyright (©) 2019. Athenas Owl. All rights reserved."
__credits__ = ["Quantiphi Analytics"]

# python dependencies
import os
import uuid
import json
import shutil
import time
import pickle

# Project related dependencies
from .constant import Constant
import zipfile

class CommonUtility:
    """
    CommonUtility is a class which provide basic functionality for Business Logic service
    """
    @staticmethod
    def timestamp_to_msec(timestamp):
        """
        timestamp_to_msec function is used to convert timestamp format into milliseconds

        @params:
            timestamp =  time in format 00:00:00.000

        @returns:
            This will return the timestamp format into (integer) milliseconds
        """
        time_split = timestamp.split(Constant.COLON)
        hour = float(time_split[0])
        minute = float(time_split[1])
        second = float(time_split[2])
        return (hour * 3600 + minute * 60 + second) * 1000

    @staticmethod
    def add_timestamp(start_time, end_time):
        """
        add_timestamp function is added to times

        @params:
            first_time  = time in 00:00:00.000 format
            second_time = time in 00:00:02.000 format

        @returns:
            This will return the addition of both the time in format 00:00:00.000
        """
        return CommonUtility.time_to_timestamp(CommonUtility.timestamp_to_msec(start_time) +
                                               CommonUtility.timestamp_to_msec(end_time))

    @staticmethod
    def time_to_timestamp(total_time_in_millisecond):
        """
        time_to_timestamp function is used to concat all the strings

        @param:
            total_time_in_millisecond = total time in milliseconds

        @returns:
            This will return the milliseconds in this time format 00:00:00.000
        """
        remaining_time_in_second, millisecond = divmod(
            total_time_in_millisecond, 1000)
        remaining_time_in_minute, second = divmod(remaining_time_in_second, 60)
        hours, minute = divmod(remaining_time_in_minute, 60)
        return Constant.TIMESTAMP_FORMAT % (hours, minute, second, millisecond)

    @staticmethod
    def concat_string(*args):
        """
        concat_string function is used to concatenate the string

        @param:
            *args = takes variable no. of arguments

        @returns:
            This will return newly created concatenated string
        """
        return "".join(map(str, args))

    @staticmethod
    def create_folder(folder_key):
        """
        create_folder function is used to take the folder key
        @param:
            folder_key = path to create folder with folder name e.g.:/tmp/sample_output/
        """
        # create a output folder if not exist
        if not os.path.exists(folder_key):
            os.makedirs(folder_key)

    @staticmethod
    def generate_uuid():
        """ generate_uuid function is used to generate uuid of 32 character

        @returns:
            This will return 32 character uuid
        """
        return (uuid.uuid4())

    @staticmethod
    def uuid_with_file_extension(extention):
        """
        uuid_with_file_extension function is used to create uuid with file extension

        @param:
            extention = takes the file extension e.g.: ".mp4"

        @returns:
            This will return filename with extension
        """
        filename = CommonUtility.concat_string(
            CommonUtility.generate_uuid(), extention)
        return filename


    @staticmethod
    def read_json_files(json_file):
        """
        read_json_files function is used to read the json

        @param:
            json_file = json file

        @returns:
            This will return json object
        """

        with open(json_file, 'r') as file:
            json_file_object = json.load(file)
        return json_file_object

    @staticmethod
    def remove_file(file):
        """
        remove_file function is used to remove the file at specified file key

        @param:
            file = file key

        @returns:
            This will return removed_status as true if file removed successfully
            else it will return false
        """
        removed_status = False
        if(os.path.isfile(file)):
            os.remove(file)
            removed_status = True

        return removed_status

    @staticmethod
    def dumps_json_in_files(json_file, json_data):
        """
        dumps_json_in_files function is used to dumps the json in file

        @param:
            json_file = json file
            json_data = json data dictionary
        """
        file_status = False
        with open(json_file, 'w') as file:
            json.dump(json_data, file)
            file_status = True
        return file_status

    @staticmethod
    def get_start_end_time(buffer_time, current_clip_duration):
        """
        Create and Save clips based on start and end
        @params:
            buffer_time = by default it is 0.2 for overlapping
            current_clip_duration = current clip duration in second

        @returns- it will return start_time and end_time
        """
        start_time = 0
        end_time = buffer_time + current_clip_duration
        return start_time, end_time

    @staticmethod
    def remove_directory(request_folder):
        """
        Remove the directory and its content
        @params:
            request_folder = Path to the folder to which we have to delete

        @returns- True
        """
        flag_status = False
        if(os.path.isdir(request_folder)):
            shutil.rmtree(request_folder)
            flag_status = True
        else:
            flag_status = False

        return flag_status

    @staticmethod
    def save_json_files(data, json_file_path):
        """
        save_json_files function is used to save the json

        @param:
            data = data to be saved in json
            json_file = json file

        @returns:
            This will return json object
        """
        with open(json_file_path, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def create_copy_of_a_file(filepath):
        """
        Create a copy of a file with uuid file name
        @params:
            filepath = Path to the file to which we have to make a copy of it

        @returns- path of a copy file
        """
        file_keys = filepath.rsplit(Constant.FILE_SEPARATOR, 1)
        file_name_key = file_keys[1].rsplit(
            Constant.FILE_EXTENSION_SEPARATOR, 1)
        copy_file_path = CommonUtility.concat_string(file_keys[0], Constant.FILE_SEPARATOR, CommonUtility.generate_uuid(
        ), Constant.FILE_EXTENSION_SEPARATOR, file_name_key[1])
        if(os.path.isfile(filepath)):
            shutil.copyfile(filepath, copy_file_path)
        return copy_file_path

    @staticmethod
    def create_copy_of_a_file_to_dest(src_filepath, dest_filepath):
        """
        Create a copy of a file
        @params:
            filepath = Path to the file to which we have to make a copy of it

        @returns- path of a copy file
        """
        if(os.path.isfile(src_filepath)):
            shutil.copyfile(src_filepath, dest_filepath)
        return dest_filepath

    @staticmethod
    def current_timestamp():
        # getting Indian time
        current_time = time.time() + 19800
        # get milliseconds
        millisec = repr(current_time).split(Constant.FILE_EXTENSION_SEPARATOR)[1][:3]
        current_time_with_millisecond = time.strftime(
            Constant.TIMESTAMP.format(millisec), time.localtime(current_time))
        return current_time_with_millisecond

    @staticmethod
    def diff_timestamp(start_time, end_time):
        """
        diff_timestamp find difference between two timestamp
        @params:
            first_time  = time in 00:00:00.000 format
            second_time = time in 00:00:02.000 format

        @returns:
            This will return the addition of both the time in format 00:00:00.000
        """
        return (CommonUtility.timestamp_to_msec(start_time) - CommonUtility.timestamp_to_msec(end_time))

    @staticmethod
    def unzip_folder(zip_file, destination_path):
        with zipfile.ZipFile(zip_file, 'r') as zipObj:
            # Extract all the contents of zip file in different directory
            zipObj.extractall(destination_path)
        return destination_path

    @staticmethod
    def write_zip(filename, source_dir):
        file_status = False
        zipf = zipfile.ZipFile(filename, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        length = len(source_dir)

        for root, dirs, files in os.walk(source_dir):
            folder = root[length:] # path without "parent"
            for file in files:
                zipf.write(os.path.join(root, file), os.path.join(folder, file))
                file_status = True
        zipf.close()
        return file_status

    @staticmethod
    def read_pickle_files(filename):
        """DocString
        """
        with open(filename , 'rb') as f:
            classifier_object = pickle.load(f)
        return classifier_object

    @staticmethod
    def roi_embedding_zip_path_finder(local_metadata_path_list):
        """DocString
        """
        unique_embedding_zip_path_dict = {}
        for local_metadata_path in local_metadata_path_list:
            data = CommonUtility.read_json_files(local_metadata_path)
            unique_zip_path = []
            for frame in data["collection"][0]["video"]["frames"]:

                for roi in frame["roi"]:
                    if roi["roi_id"]:
                        if roi["video_roi_embedding_path"] not in unique_zip_path:
                            unique_zip_path.append(roi["video_roi_embedding_path"])
                    else:
                        break
            unique_embedding_zip_path_dict[data["collection"][0]["video"]["original_video_start_time"]] = unique_zip_path

        return unique_embedding_zip_path_dict

    @staticmethod
    def is_file_exist(filepath):
        """
        Crea
        @params:
            filepath = Path to the file to which we have to be checked for existance

        @returns- flag  (True if exists else False)
        """
        flag = False
        if(os.path.isfile(filepath)):
            flag = True
        return flag
