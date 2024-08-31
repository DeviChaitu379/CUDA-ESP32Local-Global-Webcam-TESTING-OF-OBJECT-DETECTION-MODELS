###ESP VERSIOB PROGRAM FOR COCO MODELS-GPU CUDA###
##The versions of libraries and cuda , required dependencies have been mentioned in the readme file##
#note that the esp versions working (even is webcam versions work well) depends not only your gpu but the esp's image quality and the buffer speed of images
import cv2
import numpy as np
import tensorflow as tf
import urllib.request

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #>>gave 1 denoting that GPU was found
print("TensorFlow built with CUDA: ", tf.test.is_built_with_cuda())  #>>gave True denoting Tensorflow build with CUDA support

# Load the  model
model_dir = 'c:/Users/CHAITU/Documents/MATLAB/PYTHON2024/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/saved_model/'
#model_dir = 'C:/Users/CHAITU/Documents/MATLAB/PYTHON2024/trainedmodels1/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
model = tf.saved_model.load(model_dir)

# Load the labels (COCO labels)
label_map = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 
    34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 
    40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 
    70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

# Function to fetch and preprocess image
def fetch_and_preprocess_image(url):
    img_resp = urllib.request.urlopen(url)
    img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    image = cv2.imdecode(img_np, -1)
    img_resized = image  # Resize for model input
    img_resized = img_resized.astype(np.float32)
    img_resized = np.expand_dims(img_resized, axis=0)
    return image, img_resized

# Function to draw bounding boxes and labels
def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    h, w, _ = image.shape
    for i in range(boxes.shape[0]):
        if scores[i] > threshold:
            box = boxes[i] * np.array([h, w, h, w])
            (startY, startX, endY, endX) = box.astype("int")
            label = f"{label_map.get(classes[i], 'Unknown')} : {scores[i]:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Continuously fetch images from the ESP32-CAM and run inference
url = 'http://192.168.1.100/cam-hix.jpg'

while True:
    original_image, input_tensor = fetch_and_preprocess_image(url)

    detections = model(input_tensor)

    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)
    scores = detections['detection_scores'][0].numpy()

    # Draw bounding boxes on the og
    draw_boxes(original_image, boxes, classes, scores)

    # Display
    cv2.imshow("Object Detection", original_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
