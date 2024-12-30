import numpy as np
import pandas as pd
import pickle
import os
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import xgboost

with open('xgb_cv3.sav', 'rb') as to_read:
    xgb_opt = pickle.load(to_read)


def predict_result(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0], 256, symmetric=True, normed=True)
    test =  pd.DataFrame({"variance":[np.var(gray_image)],"std":[np.std(gray_image)],'mean':[np.mean(gray_image)],'contrast':[graycoprops(glcm, 'contrast')[0, 0]],'correlation':[graycoprops(glcm, 'correlation')[0, 0]],'energy':[graycoprops(glcm, 'energy')[0, 0]],'homogeneity':[graycoprops(glcm, 'homogeneity')[0, 0]],'entropy':[shannon_entropy(gray_image)]})
    result = xgb_opt.predict(test)
    return result

def predict_blank(image_path):
    convert_path = os.path.normpath(image_path)
    new_folder = f'{convert_path}\\blank'
    os.makedirs(new_folder,exist_ok=True)

    images = os.listdir(convert_path)

    for image in images:
         img_path = os.path.join(convert_path, image)
         if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(img_path)
            if predict_result(img) == 1:
                os.replace(img_path,f'{new_folder}\\{image}')

if __name__ == "__main__":
    user_input_path = input("Enter the path to the directory containing images: ")
    predict_blank(user_input_path)

