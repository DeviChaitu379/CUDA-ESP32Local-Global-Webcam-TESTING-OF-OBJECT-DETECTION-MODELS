###WEBCAM VERSIOB PROGRAM FOR COCO MODELS-GPU CUDA###
##The versions of libraries and cuda , required dependencies have been mentioned in the readme file##
import cv2
import numpy as np
import tensorflow as tf

#making sure our code works on gpu by checking tf2 build with cuda
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #>>gave 1 denoting that GPU was found
print("TensorFlow built with CUDA: ", tf.test.is_built_with_cuda())  #>>gave True denoting Tensorflow build with CUDA support

# Load the SSD MobileNet model 
model_dir = 'c:/Users/CHAITU/Documents/MATLAB/PYTHON2024/trainedmodels1/efficientdet_d3_coco17_tpu-32/saved_model'
model = tf.saved_model.load(model_dir)

# Load the labels , I used coco labels here which only is compatible with coco models
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

# Function to capture video from webcam
def capture_from_webcam():
  cap = cv2.VideoCapture(0)  # 0 for default , or you can use 1 for external ones
  while True:
    ret, frame = cap.read()
    if not ret:
      print("Error: Unable to capture frame from webcam")
      break
    yield frame
  cap.release()
  cv2.destroyAllWindows()

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

# Main loop for object detection
for frame in capture_from_webcam():
  # Preprocess the frame (convert to RGB and resize based on model input)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(frame_rgb, (640, 640))  # Adjust if model input size is different
  img_resized = img_resized.astype(np.float32)
  img_resized = np.expand_dims(img_resized, axis=0)

  # Run inference
  detections = model(img_resized)

  # Extract detection results
  boxes = detections['detection_boxes'][0].numpy()
  classes = detections['detection_classes'][0].numpy().astype(np.int32)
  scores = detections['detection_scores'][0].numpy()

  # Draw bounding boxes on the frame
  draw_boxes(frame, boxes, classes, scores)

  # Display the output
  cv2.imshow("Object Detection (Webcam)", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
