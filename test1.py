import cv2
import numpy as np
import os
import time
import pandas as pd

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)

total_execution_time = 0

def draw_label(input_image, label, left, top):
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    return outputs


def post_process(input_image, outputs, classes):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []

    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non-maximum suppression to eliminate redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detected_objects = len(indices)

    objects_counts = {}  # Dictionary to store object counts

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i]*100)
        draw_label(input_image, label, left, top)
        print(label)

        object_name = classes[class_ids[i]]
        if object_name in objects_counts:
            objects_counts[object_name] += 1
        else:
            objects_counts[object_name] = 1

    return input_image, detected_objects, objects_counts


def process_folder(input_folder, output_folder):
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the weight files to the model and load the network using them.
    model_name = "yolov5"
    modelWeights = "models/yolov5m.onnx"
    net = cv2.dnn.readNet(modelWeights)

    # Get the list of files in the input folder.
    files = os.listdir(input_folder)

    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    global total_execution_time
    total_execution_time = 0

    total_objects_detected = 0

    # Lists to store results
    image_paths = []
    model_names = []
    object_counts = []
    total_objects_detected_list = []
    execution_times = []
    objects_counts_list = []

    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):

            image_path = os.path.join(input_folder, file)
            frame = cv2.imread(image_path)

            start_time = time.time()
            detections = pre_process(frame, net)
            img, objects_detected, objects_counts = post_process(frame.copy(), detections, classes)
            end_time = time.time()

            execution_time = end_time - start_time
            total_execution_time += execution_time

            total_objects_detected += objects_detected

            # Put efficiency information.
            t, _ = net.getPerfProfile()
            label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
            draw_label(img, label, 10, 10)

            # Save the output image in the output folder.
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, img)

            print("Saved output image:", output_path)
            print("Execution time:", execution_time, "seconds")

            # Print object counts
            print("Object counts:")
            for object_name, count in objects_counts.items():
                print(f"{object_name}: {count}")

            print("\n")

            # Append results to lists
            image_paths.append(image_path)
            model_names.append(model_name)
            object_counts.append(len(objects_counts))
            total_objects_detected_list.append(total_objects_detected)
            execution_times.append(execution_time)
            objects_counts_list.append(objects_counts)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Model Name': model_names,
        'Object Counts': objects_counts_list,
        'Total Objects Detected': total_objects_detected_list,
        'Execution Time (seconds)': execution_times
    })

    # Save the DataFrame to an Excel file
    output_excel_path = os.path.join(output_folder, 'object_detection_results.xlsx')
    results_df.to_excel(output_excel_path, index=False)

    print("All images processed.")
    print("Model Name:", model_name)
    print("Total objects detected:", total_objects_detected)
    print("Total execution time for all images:", total_execution_time, "seconds")
    print("Results saved to:", output_excel_path)


if __name__ == '__main__':
    input_folder = "C:/Users/akhil/Desktop/v5/input_img2"
    output_folder = "C:/Users/akhil/Desktop/v5/output_img2"

    process_folder(input_folder, output_folder)
