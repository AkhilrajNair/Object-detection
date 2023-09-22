import cv2
import base64
import numpy as np
import api_yolov5_function as func
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/predict", methods=["POST"])
def predict():
    image_data = request.json["image"]
    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    modelWeights = "models/yolov5m.onnx"
    net = cv2.dnn.readNet(modelWeights)

    detections = func.pre_process(image, net)
    detected_objects, confidence_percentages, output_image = func.post_process(image.copy(), detections)

    response = {
        "detected_objects": detected_objects,
        "confidence_percentages": confidence_percentages,
        # "height": output_image.shape[0],
        # "width": output_image.shape[1],
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()
