from roboflow import Roboflow
from ultralytics import YOLO
from IPython.display import display, Image
import cv2
import pytesseract
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import re
from google.colab import files
import numpy as np
from PIL import Image as PILImage

rf = Roboflow(api_key="PaIk6y4dQo19iKXiPEzC")
project = rf.workspace("elsoudy").project("card-hleyg")
version = project.version(1)
dataset = version.download("yolov8", "datasets/card-1")


from ultralytics import YOLO

model = YOLO('"/content/runs/detect/train/weights/best.pt"')
results = model.val()

print(f"Precision: {results.box.mp:.4f}")   
print(f"Recall: {results.box.mr:.4f}")     
print(f"mAP@50: {results.box.map50:.4f}")  
print(f"mAP@50-95: {results.box.map:.4f}")  

from roboflow import Roboflow
rf = Roboflow(api_key="PaIk6y4dQo19iKXiPEzC")
project = rf.workspace("egyptian-ids").project("arabic-numbers-vmdt0")
version = project.version(2)
dataset = version.download("yolov8", "datasets/arabic-numbers-2")

 

from ultralytics import YOLO

 
model = YOLO('"/content/runs/detect/train2/weights/best.pt"')
 
results = model.val()

 
print(f"Precision: {results.box.mp:.4f}")   
print(f"Recall: {results.box.mr:.4f}")     
print(f"mAP@50: {results.box.map50:.4f}")   
print(f"mAP@50-95: {results.box.map:.4f}")  


from roboflow import Roboflow
rf = Roboflow(api_key="PaIk6y4dQo19iKXiPEzC")
project = rf.workspace("iddetection-zr0sa").project("national-id-ltfb6")
version = project.version(7)
dataset = version.download("yolov8", "datasets/national-id-7")

 

from ultralytics import YOLO

 
model = YOLO('"/content/runs/detect/train3/weights/best.pt"')
 
results = model.val()

 
print(f"Precision: {results.box.mp:.4f}" )   
print(f"Recall: {results.box.mr:.4f}")     
print(f"mAP@50: {results.box.map50:.4f}")  
print(f"mAP@50-95: {results.box.map:.4f}")  

 

 
def preprocess_image(cropped_image):
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary

 
def extract_text(image, bbox, lang='ara'):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    preprocessed_image = preprocess_image(cropped_image)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_image, lang=lang, config=custom_config)
    return text.strip()

 
def detect_national_id(cropped_image):
    model = YOLO('/content/detect_arabic_numbers.pt')  # Load the model directly in the function
    results = model(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_info.append((cls, x1))
            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cropped_image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    detected_info.sort(key=lambda x: x[1])
    id_number = ''.join([str(cls) for cls, _ in detected_info])

    cv2_imshow(cropped_image)
    return id_number

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def plot_image_with_boxes(image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()

def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]

def process_image(cropped_image):
    model = YOLO('/content/detect_info.pt')
    results = model(cropped_image)

    first_name = ''
    second_name = ''
    merged_name = ''
    nid = ''
    address1 = ''
    address2 = ''

    for result in results:
    

        boxes = [box.xyxy[0].tolist() for box in result.boxes]
        plot_image_with_boxes(cropped_image, boxes)

        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            bbox = [int(coord) for coord in bbox]

            if class_name == 'FN':
                first_name = extract_text(cropped_image, bbox, lang='ara')
            elif class_name == 'LN':
                second_name = extract_text(cropped_image, bbox, lang='ara')
            elif class_name == 'Add1':
                address1 = extract_text(cropped_image, bbox, lang='ara')
                address1 = remove_numbers(address1)
            elif class_name == 'Add2':
                address2 = extract_text(cropped_image, bbox, lang='ara')
                address2 = remove_numbers(address2)
            elif class_name == 'Id':
          
                nid = detect_national_id(cropped_image)

    merged_name = f"{first_name} {second_name}"
    print(f"First Name: {first_name}")
    print(f"Second Name: {second_name}")
    print(f"Full Name: {merged_name}")
    print(f"National ID: {nid}")
    print(f"Address1: {address1}")
    print(f"Address2: {address2}")

    decoded_info = decode_egyptian_id(nid)
    for key, value in decoded_info.items():
        print(f"{key}: {value}")

def decode_egyptian_id(id_number):
    governorates = {
        '01': 'Cairo',
        '02': 'Alexandria',
        '03': 'Port Said',
        '04': 'Suez',
        '11': 'Damietta',
        '12': 'Dakahlia',
        '13': 'Ash Sharqia',
        '14': 'Kaliobeya',
        '15': 'Kafr El - Sheikh',
        '16': 'Gharbia',
        '17': 'Monoufia',
        '18': 'El Beheira',
        '19': 'Ismailia',
        '21': 'Giza',
        '22': 'Beni Suef',
        '23': 'Fayoum',
        '24': 'El Menia',
        '25': 'Assiut',
        '26': 'Sohag',
        '27': 'Qena',
        '28': 'Aswan',
        '29': 'Luxor',
        '31': 'Red Sea',
        '32': 'New Valley',
        '33': 'Matrouh',
        '34': 'North Sinai',
        '35': 'South Sinai',
        '88': 'Foreign'
    }

    if len(id_number) != 14:
        raise ValueError("ID number must be 14 digits long")

    century_digit = int(id_number[0])
    year = int(id_number[1:3])
    month = int(id_number[3:5])
    day = int(id_number[5:7])
    governorate_code = id_number[7:9]
    gender_code = int(id_number[12:13])

    if century_digit == 2:
        century = "1900-1999"
        full_year = 1900 + year
    elif century_digit == 3:
        century = "2000-2099"
        full_year = 2000 + year
    else:
        raise ValueError("Invalid century digit")

    gender = "Male" if gender_code % 2 != 0 else "Female"
    governorate = governorates.get(governorate_code, "Unknown")
    birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

    return {
        'Birth Date': birth_date,
        'Governorate': governorate,
        'Gender': gender
    }

def process_detected_image(image_path):
    model = YOLO('/content/detect_id_card.pt')

    results = model(image_path)

    img = PILImage.open(image_path)

    rotated_img = img.copy() 

    for result in results:
        for box in result.boxes:
            class_name = result.names[box.cls[0].item()]
            print(f"Detected Class: {class_name}")

            if class_name == "front-right":
                rotated_img = img.rotate(-90, expand=True)  
                print(f"Rotated image 90 degrees to the right for class: {class_name}")
            elif class_name == "front-left":
                rotated_img = img.rotate(90, expand=True) 
                print(f"Rotated image 90 degrees to the left for class: {class_name}")
            elif class_name == "front-bottom":
                rotated_img = img.rotate(180, expand=True) 
                print(f"Rotated image 180 degrees for class: {class_name}")
            elif class_name == "front-up":
                rotated_img = img.rotate(0, expand=True)
                print(f"image doesn't need to rotate: {class_name}")

    rotated_image_cv2 = cv2.cvtColor(np.array(rotated_img), cv2.COLOR_RGB2BGR)

    display(rotated_img)

    return rotated_image_cv2

def detect_and_process_id_card(image_path):
    rotated_image = process_detected_image(image_path)

    id_card_model = YOLO('/content/detect_id_card.pt')
    id_card_results = id_card_model(rotated_image)

    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])   
            cropped_image = rotated_image[y1:y2, x1:x2]

    process_image(cropped_image)

uploaded = files.upload() 

for filename in uploaded.keys():
    detect_and_process_id_card(filename)  
 
model_job = YOLO('/content/detect_info.pt') 

def extract_text(image, bbox, lang='ara'):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]

    cv2_imshow(cropped_image)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(cropped_image, lang=lang, config=custom_config)
    return text.strip()

def plot_image_with_boxes(image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax = plt.gca()
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()

def detect_id_card(image_path, model):
    results = model(image_path)
    image = cv2.imread(image_path)

    id_card_bboxes = []
    for result in results:
        id_card_bboxes.extend([box.xyxy[0].tolist() for box in result.boxes])

    return image, id_card_bboxes

def process_image(image_path, model_id_card, model_job):
    image, id_card_bboxes = detect_id_card(image_path, model_id_card)

    if id_card_bboxes:
        bbox = id_card_bboxes[0]

        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        id_card_image = image[y1:y2, x1:x2]

        results = model_job(id_card_image) 

        job = ''
        for result in results:
            boxes = [box.xyxy[0].tolist() for box in result.boxes]
            plot_image_with_boxes(id_card_image, boxes) 

            for box in result.boxes:
                bbox = box.xyxy[0].tolist()  
                class_id = int(box.cls[0].item()) 
                class_name = result.names[class_id]

                bbox = [int(coord) for coord in bbox]

                if class_name == 'Job' or class_name == 'Job2':
                    job = extract_text(id_card_image, bbox, lang='ara')

        print(f"Job: {job}")
    else:
        print("No ID card detected")

uploaded = files.upload() 
for filename in uploaded.keys():
    process_image(filename, model_id_card, model_job)   