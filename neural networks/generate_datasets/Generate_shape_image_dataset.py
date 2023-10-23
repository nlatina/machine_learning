import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

# Create a directory to save the images
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    
def generate_triangle_image(n=25, value = 255):
    ''' 
    Generates an image of a triangle with random vertex placements.
    
    Parameters:
        n (int, optional): Side length of the square image. Defaults to 25.
        value (int, optional): Grayscale value to draw the image in. Defaults to 255.

    Returns:
        numpy.ndarray: An image of a randomly placed triangle within specified constraints.
    '''

    img = np.zeros((n, n), dtype=np.uint8)
    
    top_vertex = (np.random.randint(n * 8 // 25, n * 18 // 25), np.random.randint(0, n * 8 // 25))
    bottom_left_vertex = (np.random.randint(0, n * 12 // 25), np.random.randint(n * 13 // 25, n))
    bottom_right_vertex = (np.random.randint(n * 13 // 25, n), np.random.randint(n * 13 // 25, n))
    
    triangle = np.array([top_vertex, bottom_left_vertex, bottom_right_vertex])
    cv2.drawContours(img, [triangle], 0, value, -1)
    
    return img

def generate_rectangle_image(n=25, value = 255):
    ''' 
    Generates an image of a rectangle with random vertex placements.
    
    Parameters:
        n (int, optional): Side length of the square image. Defaults to 25.
        value (int, optional): Grayscale value to draw the image in. Defaults to 255.

    Returns:
        numpy.ndarray: An image of a randomly placed rectangle within specified constraints.
    '''
        
    img = np.zeros((n, n), dtype=np.uint8)
    
    top_left = (np.random.randint(n // 25, n * 10 // 25), np.random.randint(n // 25, n * 10 // 25))
    bottom_right = (np.random.randint(n * 15 // 25, n * 24 // 25), np.random.randint(n * 15 // 25, n * 24 // 25))
    
    rectangle = np.array([top_left, (bottom_right[0], top_left[1]), bottom_right, (top_left[0], bottom_right[1])])
    cv2.drawContours(img, [rectangle], 0, value, -1)
    
    return img

def generate_pentagon_image(n=25, value = 255):
    ''' 
    Generates an image of a pentagon with random vertex placements.
    
    Parameters:
        n (int, optional): Side length of the square image. Defaults to 25.
        value (int, optional): Grayscale value to draw the image in. Defaults to 255.

    Returns:
        numpy.ndarray: An image of a randomly placed pentagon within specified constraints.
    '''
        
    img = np.zeros((n, n), dtype=np.uint8)
    
    center = (n // 2 + np.random.uniform(-n * 2 // 25, n * 2 // 25), n // 2 + np.random.uniform(-n * 2 // 25, n * 2 // 25))
    radius = np.random.randint(n * 8 // 25, n * 13 // 25)
    
    base_angle = 2 * np.pi / 5
    angles = [i * base_angle + np.random.uniform(-0.4, 0.4) for i in range(5)]
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    pentagon = np.array(list(zip(x, y)), dtype=np.int32)
    cv2.drawContours(img, [pentagon], 0, value, -1)
        
    return img

### generate the dataset and save as csv ###

num_images = 2000 # number of images per shape

imgs = generate_triangle_image(n = 25).flatten() #first image
labels = [0]

for i in range(num_images):
    img1 = generate_triangle_image(n = 25).flatten()
    labels.append(0)
    
    img2 = generate_rectangle_image(n = 25).flatten()
    labels.append(1)
    
    img3 = generate_pentagon_image(n = 25).flatten()
    labels.append(2)

    
    imgs = np.c_[imgs, img1, img2, img3]
    
    
print("Dataset generated!")

dataset = np.c_[labels, imgs.T]
cols = ['label']
for i in range(625):
    cols.append(f"pixel_{i}")
dataset = pd.DataFrame(dataset, columns = cols)
dataset.to_csv("shape_set_3-5.csv")