import cv2
from pyzbar.pyzbar import decode
import os
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image
from io import BytesIO
import random
import glob
import numpy as np
import time
from tensorflow.lite.python.interpreter import Interpreter
import cv2
import pytesseract
import re
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from playsound import playsound
from datetime import datetime
from flask_cors import CORS  # Import COR
from pyzbar.pyzbar import decode


cap = cv2.VideoCapture(0)

camera = True

while camera:
    _, frame = cap.read()
    for barcode in decode(frame):
        print(barcode.data)
        myData = barcode.data.decode('utf-8')
        print(myData)

    cv2.imshow('Result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# AS_found_count = [0]
# found_match_count = [0]



# df = pd.read_csv('/Users/edgaryepez/AmericanShip/AmericanShip2/tflite1/objdetection/CustomersView.csv')

# # Create a dictionary with client ID as key and a dictionary of names as value
# customer_data = {}
# for index, row in df.iterrows():
#     client_id = row['Unique STE # AS-']
#     customer_data[client_id] = {
#         'first_name': row['First Name'],
#         'last_name': row['Last Name']
#     }

# def matching(image):

#     # Set the Tesseract executable path
#     os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/'
#     pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


#     # Perform OCR using the custom model
#     custom_config = r'--tessdata-dir "/opt/homebrew/share/tessdata" -l box3'
#     text = pytesseract.image_to_string(image, config=custom_config)
#     print("this is text")
#     print(text)

#     pattern = re.compile(r'AS\d{5}')
#     match = pattern.search(text)

#     if match:
#         found_as = match.group()
#         found_as = int(found_as[2:])

#         if found_as in customer_data:
#             first_name = customer_data[found_as]['first_name']
#             last_name = customer_data[found_as]['last_name']
            
#             # Create a regex pattern to search for the names
#             pattern2 = rf"{first_name} "
#             pattern3 = rf"{last_name}"

#             # Check if the names are found in the text
#             if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
#                 print(f"Found: {first_name} {last_name} with ID {found_as}")

                
            
#                 playsound(r'/Users/edgaryepez/AmericanShip/AmericanShip2/tflite1/Correct_Answer_sound_effect.mp3')
                
#                 return True, first_name, last_name, found_as

#     return False, None, None, None
    

# def convert_to_pil(image):
#     if isinstance(image, np.ndarray):
#         return Image.fromarray(image)
#     elif isinstance(image, Image.Image):
#         return image
#     else:
#         raise TypeError("The image must be a NumPy array or PIL Image.")
    



# def deskew(image_path):
#     # Read the image
    
#     # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     image = image_path
#     if image is None:
#         raise ValueError("Image not found or unable to open")

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Binary thresholding
#     _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#     # Detect edges
#     edges = cv2.Canny(binary, 50, 150, apertureSize=3)

#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

#     # Check if any lines are detected
#     if lines is None:
#         # print("No lines detected")
#         return image

#     # Calculate the angle of rotation based on the detected lines
#     angles = []
#     for line in lines:
#         for rho, theta in line:
#             angle = (theta * 180 / np.pi) - 90
#             if -45 < angle < 45:  # Ensure the angle is within a reasonable range
#                 angles.append(angle)
    
#     # If no angles are within range, return the original image
#     if len(angles) == 0:
#         # print("No valid angles detected")
#         return image

#     # Calculate the median angle to avoid extreme outliers
#     median_angle = np.median(angles)

#     # Rotate the image to deskew
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     return rotated


# def enhance_image(image):
#     new_size = (image.shape[1] * 1, image.shape[0] * 1)

#     # Resize the image using OpenCV
#     enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
#     if isinstance(enlarged_image, np.ndarray):
#         image = Image.fromarray(enlarged_image)

#     image = image.convert('L')

#     enhancer = ImageEnhance.Contrast(image)
#     image = enhancer.enhance(1.5)

#     image = image.filter(ImageFilter.SHARPEN)

#     return image


# def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.5, num_test_images=10, savepath='/content/results', txt_only=False):

#     # Grab filenames of all images in test folder

#     try:
#         imgpath = cv2.imread(imgpath)  
#     except:
#         imgpath = imgpath

#     matchfound = False
    
#     # Load the label map into memory
#     with open(lblpath, 'r') as f:
#         labels = [line.strip() for line in f.readlines()]

#     # Load the Tensorflow Lite model into memory
#     interpreter = Interpreter(model_path=modelpath)
#     interpreter.allocate_tensors()

#     # Get model details
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     height = input_details[0]['shape'][1]
#     width = input_details[0]['shape'][2]

#     float_input = (input_details[0]['dtype'] == np.float32)

#     input_mean = 127.5
#     input_std = 127.5

#     # Randomly select test images
    

#     # Loop over every image and perform detection
  

#     # Load image and resize to expected shape [1xHxWx3]
#     image = deskew(imgpath)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     imH, imW, _ = image.shape
#     image_resized = cv2.resize(image_rgb, (width, height))
#     input_data = np.expand_dims(image_resized, axis=0)

#     # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
#     if float_input:
#         input_data = (np.float32(input_data) - input_mean) / input_std

#     # Perform the actual detection by running the model with the image as input
#     interpreter.set_tensor(input_details[0]['index'],input_data)
#     interpreter.invoke()

#     # Retrieve detection results
#     boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
#     classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
#     scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

#     detections = []

#     # Loop over all detections and draw detection box if confidence is above minimum threshold
#     print("decoing")

#     image = enhance_image(image)
#     image = np.array(image)

#     plt.imshow(image)
#     plt.show()

#     for code in decode(image):
#         print(code.type)
#         print(code.data.decode('utf-8'))


#     for i in range(len(scores)):
#         if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

#             # Get bounding box coordinates and draw box
#             # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
#             ymin = int(max(1,(boxes[i][0] * imH)))
#             xmin = int(max(1,(boxes[i][1] * imW)))
#             ymax = int(min(imH,(boxes[i][2] * imH)))
#             xmax = int(min(imW,(boxes[i][3] * imW)))

#             cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
#         #   plt.imshow(image)
#         #   plt.show()

#             # Crop Image
            
#             crop_img = image[ymin:ymax, xmin:xmax]


        

#             matchfound, first_name, last_name, As = matching(crop_img)


#             return matchfound, first_name, last_name, As

#             if matchfound == False:
#                 print("no match found")
#             #   os.makedirs('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_To_Be_Trained', exist_ok=True)
#             #   savepath = os.path.join('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_CannotRead',os.path.basename(image_path))
#             #   print(savepath) 
#             #   enhanced_image.save(savepath)
#         #     enhanced_image = np.array(enhanced_image)
#         #     text =easyOCR(enhanced_image)
#         #     __, matchfound = found_AS_and_Match(enhanced_image)


                


#             # Draw label
#             object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
#             label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
#             labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
#             label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
#             cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
#             cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

#             detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

#     if detections == []:
#         print('No detections found.')
#         # os.makedirs('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_No_detections', exist_ok=True)
#         # savepath = os.path.join('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_No_detections',os.path.basename(image_path))
#         # print(savepath)
#         # newimage = convert_to_pil(image)
#         # newimage.save(savepath)



#     return matchfound, "", "", ""

# import cv2

# # Open a connection to the webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# print("Press the spacebar to take a picture, or 'q' to quit.")

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Display the resulting frame
#     cv2.imshow('Live Feed', frame)

#     # Wait for a key press
#     key = cv2.waitKey(1) & 0xFF

#     # If spacebar is pressed, save the frame
#     if key == ord(' '):
#         image = frame
#         print("Picture taken!")
#         break
#     # If 'q' is pressed, quit without taking a picture
#     elif key == ord('q'):
#         print("Quitting without taking a picture.")
#         image = None
#         break

# # Release the webcam and close the window
# cap.release()

# matchfound, first_name, last_name, As = tflite_detect_images(
#         modelpath="/Users/edgaryepez/AmericanShip/AmericanShip2/tflite1/custom_model_liteBOX/detect.tflite",
#         imgpath=image,
#         lblpath="/Users/edgaryepez/AmericanShip/AmericanShip2/tflite1/custom_model_liteBOX/labelmap.txt",
#         min_conf=0.1,
#         num_test_images=1,
#     )