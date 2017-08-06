import numpy as np
import cv2
import csv
import random
import sklearn

def generator(metadata, create_image=True, batch_size=128, non_center_image_correction_value=0.25):

    # Locally 
    # IMG_BASE_LOCATION = "/Users/mentlsve/dev/SDC/CarND-Behavioral-Cloning-P3/data/IMG/"
    # On AWS
    IMG_BASE_LOCATION = "/home/carnd/data/IMG/"
    
    num_samples = len(metadata)

    sklearn.utils.shuffle(metadata)
    idx = 0
    while True:
        images = []
        steerings = []
        while len(steerings) < batch_size:
                
            batch_sample = metadata[idx]
            camera_idx = random.randint(0, 2)

            # construction of image_path
            image_path = IMG_BASE_LOCATION + batch_sample[camera_idx].split('/')[-1]
            
            # steering value calculation
            center_steering = float(batch_sample[3])
            steering = center_steering
            if camera_idx == 1: #left
                steering = steering + non_center_image_correction_value
            if camera_idx == 2: #right
                steering = steering - non_center_image_correction_value

            #if create_image:
            if create_image:
                images, steerings = append_based_on_steering_value(image_path, steering, center_steering, images, steerings)
            else:
                steerings = append_based_on_steering_value_steering_value_only(steering, center_steering, steerings)
            idx = (idx + 1) % (num_samples)
            
        yield np.array(images), np.array(steerings)

def append_based_on_steering_value(image, steering, center_steering, images, steerings):
    rnd_val = random.random()
    if abs(center_steering) < 0.1 and rnd_val < 0.70:
        pass
    else:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        if rnd_val < 0.5:
            steerings.append(steering)
            images.append(image)
        else:
            steerings.append(-steering)
            images.append(np.fliplr(image))

    return images, steerings

# this method is used for creating a histogram of steering angles only
def append_based_on_steering_value_steering_value_only(steering, center_steering, steerings):
    rnd_val = random.random()
    if abs(center_steering) < 0.01 and rnd_val < 0.88:
        pass
    else:
        if rnd_val < 0.5:
            steerings.append(steering)
        else:
            steerings.append(-steering)
    return steerings

def read_dataset_metadata(path="./data/driving_log.csv"):

    lines = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines