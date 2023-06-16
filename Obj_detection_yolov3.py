import cv2
import numpy as np
import matplotlib.pyplot as plt

config_file = 'yolov3.cfg'
frozen_model = 'yolov3.weights'

yolo = cv2.dnn.readNetFromDarknet(config_file, frozen_model)
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classLabel = []
Object_file = 'Labels.txt'
with open(Object_file, 'rt') as fpt:
    classLabel = fpt.read().rstrip('\n').split('\n')

img = cv2.imread('C:/Users/akhil/Desktop/imagecv/input_img/in4.jpeg')

blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
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
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, width, height])

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN

for i in indices.flatten():
    x, y, width, height = boxes[i]
    label = f"{classLabel[class_ids[i]].upper()} {confidences[i] * 100:.2f}%"
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 10), font, fontScale=font_scale, color=(255, 0, 0), thickness=2)
    print(label)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imshow("result Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
