import os
import shutil

original_path = "/Users/sarahmac/Documents/Top up/Information Technology Project/Python program/PythonServer"

def moveFile(username):
    data_folder = os.path.join(original_path, 'data')
    selected_data_folder = os.path.join(original_path, 'selectedData')

    user_number = username[4:]
    user_data = f'S{user_number}.pkl'

    for file_name in os.listdir(data_folder):
        source_path = os.path.join(data_folder, file_name)
        if (file_name == user_data):
            shutil.move(source_path, os.path.join(selected_data_folder, file_name))

    for file_name in os.listdir(selected_data_folder):
        source_path = os.path.join(selected_data_folder, file_name)
        if(file_name != user_data):
            shutil.move(source_path, os.path.join(data_folder, file_name))