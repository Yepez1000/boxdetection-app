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



app = Flask(__name__)
CORS(app)

AS_found_count = [0]
found_match_count = [0]



df = pd.read_csv('CustomersView.csv')

# Create a dictionary with client ID as key and a dictionary of names as value
customer_data = {}
for index, row in df.iterrows():
    client_id = row['Unique STE # AS-']
    customer_data[client_id] = {
        'first_name': row['First Name'],
        'last_name': row['Last Name']
    }

def process_tracking_number(tracking_number):
    def replace_all(s, old, new):
        return s.replace(old, new)

    def get_alpha(s):
        return ''.join(filter(str.isalpha, s))

    # Initialize result dictionary
    result = {
        "Tracking": tracking_number,
        "Carrier": "",
        "Tracking_Message": ""
    }

    track_length = len(tracking_number)

    if track_length == 34:
        usps = tracking_number[:5]
        if usps == "42078":
            result["Tracking"] = tracking_number[8:34]
            result["Carrier"] = "USPS"
        else:
            # FEDEX
            result["Tracking"] = tracking_number[22:34]
            result["Carrier"] = "FedEx"
    elif track_length == 12:
        # FEDEX
        if tracking_number[:2] != "42" and int(tracking_number) > 700000000000:
            result["Carrier"] = "FedEx"
        

    elif track_length == 10:
        # DHL
        result["Carrier"] = "DHL"
    elif track_length == 29:
        # USPS
        result["Tracking"] = "Error"
        # You can add a more specific message here if needed
        # result["Tracking"] = "Error: Code not recognized"
    elif track_length == 30:
        # USPS
        result["Tracking"] = tracking_number[8:30]
        result["Carrier"] = "USPS"
    elif track_length == 31:
        # USPS
        tracking_number = replace_all(tracking_number, "↔", "")
        result["Tracking"] = tracking_number[8:30]
        result["Carrier"] = "USPS"
    elif track_length == 22:
        # USPS, Estafeta
        if "A" in tracking_number:
            result["Tracking"] = tracking_number
            result["Carrier"] = "Estafeta"
        elif len(get_alpha(tracking_number)) > 0:
            result["Tracking"] = "Error: Code not recognized"
        else:
            result["Carrier"] = "USPS"
    elif track_length == 18:
        if tracking_number[:2] == "1Z":
            result["Carrier"] = "UPS"
        else:
            result["Tracking"] = "Error: Code not recognized"
            result["Carrier"] = ""
    else:
        result["Tracking_Message"] = f"Bar code not recognized {tracking_number}"
        result["Tracking"] = ""

    return result






def matching(image):

    # Set the Tesseract executable path
    os.environ['TESSDATA_PREFIX'] = 'tesseract'


    # Perform OCR using the custom model
    custom_config = r'--tessdata-dir "tesseract/tessdata" -l box3'
    text = pytesseract.image_to_string(image, config=custom_config)
    print("this is text")
    print(text)

    pattern = re.compile(r'AS\d{5}')
    match = pattern.search(text)

    if match:
        found_as = match.group()
        found_as = int(found_as[2:])

        if found_as in customer_data:
            first_name = customer_data[found_as]['first_name']
            last_name = customer_data[found_as]['last_name']
            
            # Create a regex pattern to search for the names
            pattern2 = rf"{first_name} "
            pattern3 = rf"{last_name}"

            # Check if the names are found in the text
            if re.search(pattern2, text, re.IGNORECASE) or re.search(pattern3, text, re.IGNORECASE):
                print(f"Found: {first_name} {last_name} with ID {found_as}")

                
            
                playsound(r'/Users/edgaryepez/AmericanShip/tflite1/objdetection/boxdetection-app/Correct Answer Sound Effect.mp3')
                
                return True, first_name + " " + last_name, found_as

    return False, None, None
    

def convert_to_pil(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise TypeError("The image must be a NumPy array or PIL Image.")
    



def deskew(image_path):
    # Read the image
    
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = image_path
    if image is None:
        raise ValueError("Image not found or unable to open")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Check if any lines are detected
    if lines is None:
        # print("No lines detected")
        return image

    # Calculate the angle of rotation based on the detected lines
    angles = []
    for line in lines:
        for rho, theta in line:
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:  # Ensure the angle is within a reasonable range
                angles.append(angle)
    
    # If no angles are within range, return the original image
    if len(angles) == 0:
        # print("No valid angles detected")
        return image

    # Calculate the median angle to avoid extreme outliers
    median_angle = np.median(angles)

    # Rotate the image to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


def enhance_image(image):
    new_size = (image.shape[1] * 1, image.shape[0] * 1)

    # Resize the image using OpenCV
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    if isinstance(enlarged_image, np.ndarray):
        image = Image.fromarray(enlarged_image)

    image = image.convert('L')

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)

    image = image.filter(ImageFilter.SHARPEN)

    return image


def tflite_detect_images(modelpath, imgpath, lblpath, min_conf=0.1, num_test_images=10, savepath='/content/results', txt_only=False):

    # Initialize result dictionary
    result = {
        "Tracking": "",
        "Carrier": "",
        "Tracking_Message": ""
    }

    # Grab filenames of all images in test folder

    matchfound = False
    
    # Load the label map into memory
    with open(lblpath, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model into memory
    interpreter = Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    float_input = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Randomly select test images
    

    # Loop over every image and perform detection
  

    # Load image and resize to expected shape [1xHxWx3]
    image = deskew(imgpath)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    print("decoing")

    image = enhance_image(image)
    image = np.array(image)

    for code in decode(image):
        
        print(code.type)
        print(code.data.decode('utf-8'))
        tracking_number = code.data.decode('utf-8')
        result = process_tracking_number(tracking_number)
        print(result)
        if result["Carrier"] is not None:
            playsound(r'/Users/edgaryepez/AmericanShip/tflite1/Correct_Answer_sound_effect.mp3')
            break
        else:
            result = {
                "Tracking": "",
                "Carrier": "",
                "Tracking_Message": ""
            }
  


    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
        #   plt.imshow(image)
        #   plt.show()

            # Crop Image
            
            crop_img = image[ymin:ymax, xmin:xmax]


            matchfound, name, As = matching(crop_img)

            if matchfound != False:
                return matchfound, name , As, result['Tracking'], result['Carrier']


            

    if detections == []:
        print('No detections found.')
        # os.makedirs('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_No_detections', exist_ok=True)
        # savepath = os.path.join('/Users/edgaryepez/AmericanShip/AmericanShip2/2024_07_23_No_detections',os.path.basename(image_path))
        # print(savepath)
        # newimage = convert_to_pil(image)
        # newimage.save(savepath)



    return matchfound, "", "", result['Tracking'], result['Carrier']

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def serve_html():
    return send_from_directory('.', '/Users/edgaryepez/AmericanShip/tflite1/objdetection/boxdetection-app/index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/upload', methods=['POST'])
def upload_file():


    file = request.files['file']
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(file_path)



   
    image = Image.open(file.stream)
    image =np.array(image)
    matchfound, name, As, tracking_number, carrier = tflite_detect_images(
        modelpath="/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/detect.tflite",
        imgpath=image,
        lblpath="/Users/edgaryepez/AmericanShip/tflite1/custom_model_liteBOX/labelmap.txt",
        min_conf=0.01,
        num_test_images=1,
    )
   

    

    print("thjis is tracking ")
    print(tracking_number, carrier)

    return jsonify({'file_url': "hello ",'name': name,'As': As, 'tracking_number': tracking_number, 'carrier': carrier})


    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    

        

if __name__ == '__main__':
    app.run()
