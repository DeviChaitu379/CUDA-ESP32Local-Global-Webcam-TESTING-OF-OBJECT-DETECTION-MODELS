# CUDA-TESTING-OF-OBJECT-DETECTION-MODELS
This is my test to observe and analyze the working of different object detection models on CUDA cores ,downloaded from Kaggle

VERSIONS THAT ARE COMPATIBLE WITH EACHOTHER (DATE:AUGUST15th2024)

PYTHON==3.9.13
NUMPY==1.21.6 (1.x)
TENSORFLOW==2.10.1 (max compatible with win10native)(tensorflow-gpu not needed)
CUDA==11.2 or 11.8
cuDNN==8.9



      import tensorflow as tf
      print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #>>gave 1 denoting that GPU was found
      print("TensorFlow built with CUDA: ", tf.test.is_built_with_cuda())  #>>gave True denoting Tensorflow build with CUDA support



     nvidia smi (can be used in terminal cmd to check stats of gpu usage)



      import tensorflow as tf

     gpus = tf.config.list_physical_devices('GPU')
     try:
         tf.config.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU
     except RuntimeError as e:
         # Visible devices must be set before GPUs have been initialized
         print(e)






# MODELS TESTED (OBJECT DETECTION):
# model1: ssd_mobilenet_v2_coco_2018_03_29 == 
object detection tested with saved model format supporting code (2svedmodeltypeworkingimport cv2import cv2.py)
output: camvisible=Yes, green boxes, fixed label named "object



# model2: efficientdet_d0_coco17_tpu-32 == 
tested with saved model format supporting code (attampt2.1.py)
output: camvisible=yes, Blue boxes, label with name but accuracy low as it predicts cats as dogs, dogs as horses and so on..(thus incompatible          label map inside the code)
          UPDATEaug26: IT WORKS WITH NEWER CODE WHICH USES WEBCAM INSTEAD OF ESP32CAM, USES GPU CONTINOUSLY AND STABLE WITHOUT HAVING SUDDEN USAGE SPIKES.



# model 3: mobilenet_v2_1.4_224 == 
Object classification model tested with frozen.pb model type code (aug19.py) 
output: camvisible=Yes, no bixes, only object classification on top left with matching range from >0 to 1



# model 4: ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 == 
Object detction tested using the saved model type and label map inside code (ESP_COCO.py),
output: camvisible = YEs, Green boxes around the indentified objects and accuracy is above average, (more accuracy was achieved with the model           version 640x640 but is computational load is high and thus very laggy) and detects upto 90 objects as per label map and also displays the match range of >0 to 1
           UPDATEaug26: IT BECAME MORE ACCURATE WITH WEBCAM THAN WITH ESP32CAM FEED (WEBCAM_COCO.py)

           
# model 5:  efficientdet_d0_coco17_tpu-32 ==
AUG26, works same as model 2 with same code (webcam version) but is laggy and more load on gpu but is significantly more reliable than model2 in identification of object at times.



  # ##CLOUD TPU TESTS ## #

tested with esp versions of code and is working, more info will be updated
