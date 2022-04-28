from config import Config
import os
import numpy as np 
import cv2 as cv

c = Config()
file_list = os.listdir(c.train_dir)

def extract_test_cam(file_list):
    image_list = os.listdir(os.path.join(c.train_dir, file_list[0]))
    single = list(filter(lambda x: x.endswith('190.jpg'), image_list))
    test_idx = np.random.choice(single, 20)
    test_cam_list = []
    for idx in test_idx:
        test_cam = idx[:-7]
        test_cam_list.append(test_cam)
    return test_cam_list

test_cam = extract_test_cam(file_list)
print(test_cam)
for file in file_list:
    image_list = os.listdir(os.path.join(c.train_dir, file))
    test_extract = []
    for cam in test_cam:
        extract = list(filter(lambda x: x.startswith(cam), image_list))
        test_extract.extend(extract)
        for slice in extract:
            image_path = os.path.join(c.train_dir, file, slice)
            if not os.path.isfile(image_path):
                continue
            if not os.path.isdir(os.path.join(c.test_dir, file)):
                os.makedirs(os.path.join(c.test_dir, file))
            target_path = os.path.join(c.test_dir, file, slice)
            os.replace(image_path, target_path)
        
        
print("finish!")
    