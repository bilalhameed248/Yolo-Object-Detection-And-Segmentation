from django.shortcuts import render
import traceback, os
from django.conf import settings
from .forms import UploadFileForm
import cv2
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
model = YOLO("./model/yolov8m.pt")
seg_model = YOLO("yolov8m-seg.pt")
# Create your views here.

def home(request):
    try:
        print("Index")
        form = UploadFileForm()
        return render(request, 'index.html', {'form':form})
    except Exception as e:
        traceback.print_exc()
        print("Exception Home:",e)
        pass

def upload_file(request):
    try:
        if request.method == "POST":
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                file  = request.FILES['file']
                file_path = os.path.join(settings.STATIC_ROOT, 'uploaded_files', file.name)
                print("file_path:",file_path)
                with open(file_path, 'wb+') as destination:  
                    for chunk in file.chunks():
                        destination.write(chunk)
                output_file_path = detect_object_yolo(file_path)
                outlined_image_path, filled_image_path = detect_seg_yolo(file_path)
            return render(request, 'index.html', {'form':form, 'output_file_path':output_file_path})    
        else:
            form = UploadFileForm()
            return render(request, 'index.html', {'form':form})
    except Exception as e:
        traceback.print_exc()
        print("Exception Home:",e)
        pass

def detect_object_yolo(file_addess):
    try:
        image = cv2.imread(file_addess)
        result = model.predict(image)[0]
        image_height, image_width, _ = image.shape
        padding = 30
        new_height = image_height + 2 * padding
        new_width = image_width + 2 * padding
        new_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
        new_image[padding:padding+image_height, padding:padding+image_width] = image
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            x_min, y_min, x_max, y_max = cords
            cv2.rectangle(new_image, (x_min + padding, y_min + padding), (x_max + padding, y_max + padding), (0, 255, 0), 2)
            label = f"{class_id} ({conf})"
            text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = max(x_min + padding, 0)
            text_y = max(y_min - 10 + padding, text_height)
            cv2.rectangle(new_image, (text_x, text_y - text_height), (text_x + text_width, text_y), (0, 255, 0), -1)
            cv2.putText(new_image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        output_file_path = os.path.join(settings.STATIC_ROOT, 'uploaded_files/detected_image.jpg')
        cv2.imwrite(output_file_path, new_image)
        return output_file_path.replace('\\', '/')
    except Exception as e:
        traceback.print_exc()
        print("Exception detect_object:",e)
        pass


def detect_seg_yolo(file_addess):
    try:
        result = seg_model.predict(file_addess)[0]
        img_outline = Image.open(file_addess)
        img_fill = img_outline.copy()
        draw_outline = ImageDraw.Draw(img_outline)
        draw_fill = ImageDraw.Draw(img_fill)
        masks = result.masks
        outline_thickness = 5
        for mask in masks:
            polygon = mask.xy[0]

            draw_fill.polygon(polygon, fill=(0, 255, 0))

            for i in range(outline_thickness):
                offset = i - outline_thickness // 2 
                offset_polygon = [(p[0] + offset, p[1] + offset) for p in polygon]
                draw_outline.polygon(offset_polygon, outline=(0, 255, 0))

        outlined_image = np.array(img_outline)
        filled_image = np.array(img_fill)

        outlined_image_path = os.path.join(settings.STATIC_ROOT, 'uploaded_files/outlined_image.png')
        filled_image_path = os.path.join(settings.STATIC_ROOT, 'uploaded_files/filled_image.png')
        cv2.imwrite(outlined_image_path, outlined_image)
        cv2.imwrite(filled_image_path, filled_image)
        return outlined_image_path.replace('\\', '/'), filled_image_path.replace('\\', '/')
    except Exception as e:
        traceback.print_exc()
        print("Exception detect_object:",e)
        pass




def detect_object_cc(file_addess):
    try:
        image = cv2.imread(file_addess)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        #Save the detected image here:
        output_file_path = os.path.join(settings.STATIC_ROOT, 'uploaded_files/detected_image.jpg')
        cv2.imwrite(output_file_path, image)
        return output_file_path.replace('\\', '/')
    except Exception as e:
        traceback.print_exc()
        print("Exception detect_object:",e)
        pass
