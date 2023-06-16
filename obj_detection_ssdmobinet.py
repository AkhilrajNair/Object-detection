# BASIC OBJECT DETECTION CODE

import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabel = []
Object_file = 'Labels.txt'
with open(Object_file, 'rt') as fpt:
    classLabel = fpt.read().rstrip('\n').split('\n')
# print(classLabel)

# print(len(classLabel))

model.setInputSize(480,480)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

img = cv2.imread('C:/Users/akhil/Desktop/imagecv/input_img/in6.jpeg')

ClassIndex, confidence, bbox = model.detect(img, confThreshold = 0.5)
print(ClassIndex)

font_scale = 1
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    #print(boxes[1])
    cv2.rectangle(img, boxes, (0,255,0),2)
    label = f"{classLabel[ClassInd - 1].upper()} {conf * 100:.2f}%"
    cv2.putText(img, label, (boxes[0], boxes[1] + 15) , font, fontScale = font_scale , color = (255,0,0), thickness = 2)
    print(label)
    
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imshow("result Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()