import cv2
import numpy as np
import os

def remove_bubbles(image_path):
    convert_path = os.path.normpath(image_path)
    new_folder = f'{convert_path}\\bubbles'
    os.makedirs(new_folder,exist_ok=True)

    images = os.listdir(convert_path)

    for image in images:
         if image.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(convert_path, image)
            img = cv2.imread(image_path)

            # Convert the image from BGR to HSV color space
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define the range of blue color in HSV
            lower_blue = np.array([75, 50, 50])
            upper_blue = np.array([135, 255, 255])
            
            # Larger range
            #lower_blue = np.array([90, 50, 50])
            #upper_blue = np.array([130, 255, 255])

            # Smaller range
            #lower_blue = np.array([100, 150, 0])
            #upper_blue = np.array([140, 255, 255])

            # Create a mask for blue color
            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

            # Convert blue pixels to white in the original image
            img[mask != 0] = [255, 255, 255]

            # Convert to grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply a binary threshold to the grayscale image
            _, binary_image = cv2.threshold(gray_image, 60 , 255, cv2.THRESH_BINARY) 

            # Apply a Gaussian blur to reduce noise and improve circle detection
            blurred_image = cv2.GaussianBlur(binary_image, (9, 9), 2)

            # Detect circles using Hough Circle Transform
            circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                    param1=50, param2=30, minRadius=5, maxRadius=2000)

            if circles is not None: 
                os.replace(image_path,f'{new_folder}\\{image}')

if __name__ == "__main__":
    user_input_path = input("Enter the path to the directory containing images: ")
    remove_bubbles(user_input_path)

