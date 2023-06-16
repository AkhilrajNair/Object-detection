import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

models_map = {
    "yolov3": ['yolov3.cfg', 'yolov3.weights'],
    "yolov4": ['yolov4.cfg', 'yolov4.weights'],
    "yolov7": ['yolov7.cfg', 'yolov7.weights']
}

def model_load(model_name):
    config_file, frozen_model = models_map[model_name]
    yolo = cv2.dnn.readNetFromDarknet(config_file, frozen_model)
    yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return yolo

# Function to perform object detection on an image and save the result
def object_detection(image_path, output_image_folder, model_name):
    image = cv2.imread(image_path)

    yolo = model_load(model_name)

    classLabel = []
    Object_file = 'Labels.txt'
    with open(Object_file, 'rt') as fpt:
        classLabel = fpt.read().rstrip('\n').split('\n')

    blob = cv2.dnn.blobFromImage(image, 1/255, (480, 480), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)

    output_layer_names = yolo.getUnconnectedOutLayersNames()
    layer_outputs = yolo.forward(output_layer_names)

    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    if len(boxes) == 0:
        print(f"No objects found in the {input_image}.")
        # Save the input image as the output image
        output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
        cv2.imwrite(output_image_path, image)
        return {}

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)

    font_scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    objects_in_image = {}
    for i in indices.flatten():
        x, y, width, height = boxes[i]
        label = f"{classLabel[class_ids[i]].upper()}"
        percent = f"{confidences[i] * 100:.2f}%"

        if label not in objects_in_image:
            objects_in_image[label] = 1
        else:
            objects_in_image[label] += 1

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f"{label}{percent}", (x, y - 10), font, fontScale=font_scale, color=(255, 0, 0), thickness=2)
        # (f"{label} {percent}")

    # Save the output image with detected objects
    output_image_path = os.path.join(output_image_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    return objects_in_image


# Input folder
input_image_folder = "C:/Users/akhil/Desktop/imagecv/input_img"

# Output folder to save the images with detected objects
output_image_folder = "C:/Users/akhil/Desktop/imagecv/output_img"

model_names = ["yolov3","yolov4","yolov7"]

all_labels = set()  # Set to store all unique labels

data = []
for model_name in model_names:
    # Initialize total execution time and total objects count
    total_execution_time = 0
    total_objects = 0

    all_objects = {}

    for input_image in os.listdir(input_image_folder):
        if input_image.endswith(".jpg") or input_image.endswith(".jpeg") or input_image.endswith(".png"):
            image_path = os.path.join(input_image_folder, input_image)

            start_time = time.time()

            objects_in_image = object_detection(image_path, output_image_folder, model_name)

            # Add to all object dictionary
            for label, count in objects_in_image.items():
                if label not in all_objects:
                    all_objects[label] = count
                else:
                    all_objects[label] += count

                all_labels.add(label)  # Add label to the set of all labels

            execution_time = time.time() - start_time
            total_execution_time += execution_time

            total_objects += sum(objects_in_image.values())

            print(f"Execution time for {input_image}: {execution_time:.2f} seconds")

    # Append the data to the list
    data.append([model_name, all_objects, total_objects, total_execution_time])

# Create a sorted list of all unique labels
all_labels = sorted(list(all_labels))
label_columns = {label: [] for label in all_labels}

# Iterate through the data and populate the label columns
for row in data:
    all_objects = row[1]
    for label in all_labels:
        label_columns[label].append(all_objects.get(label, 0))

# Create the DataFrame with label columns
df = pd.DataFrame(data, columns=["Model Name", "Labels", "Total Detected Objects", "Total Execution Time"])
for label in all_labels:
    df[label] = label_columns[label]

# Save the DataFrame to an Excel file
df.to_excel("output_data.xlsx", index=False)
