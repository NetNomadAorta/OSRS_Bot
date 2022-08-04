import os
import cv2
import glob
import shutil
import re


# User Parameters/Constants to Set
DATA_DIR_1 = "./0-All/"
DATA_DIR_2 = "./0-cont_missh_only/"
KEEP_ONLY_DEFECTS = False # Will only keep images of defects in AA_NoDefect folder or first folder


def deleteDirContents(dir):
    # Deletes photos in path "dir"
    # # Used for deleting previous cropped photos from last run
    for f in os.listdir(dir):
        fullName = os.path.join(dir, f)
        shutil.rmtree(fullName)



# Main()
i=0
for image_name_1 in os.listdir(DATA_DIR_1):
    
    if '_annotations.coco.json' in image_name_1:
        continue
    
    has_match = False
    
    splitted_name = image_name_1.split(".rf")[0]
    
    for image_name_2 in os.listdir(DATA_DIR_2):
        
        if '_annotations.coco.json' in image_name_2:
            continue
        
        if splitted_name in image_name_2:
            has_match = True
    
    if has_match:
        i += 1
    else:
        image_path_1 = os.path.join(DATA_DIR_1, image_name_1)
        os.remove(image_path_1)
    
print(i)