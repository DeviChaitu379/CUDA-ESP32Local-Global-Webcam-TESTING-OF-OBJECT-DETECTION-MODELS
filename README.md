# CUDA-TESTING-OF-OBJECT-DETECTION-MODELS-ON ESP32 Local/Global and Webcam

Welcome to our project where we test the capabilities of various object detection models on CUDA cores using an ESP32 camera. This repository contains a collection of Python scripts that demonstrate how different object detection models perform on this powerful embedded platform.

**Compatible Versions:**
- **Python:** 3.9.13
- **NumPy:** 1.21.6 (1.x)
- **TensorFlow:** 2.10.1 (max compatible with win10native)(tensorflow-gpu not needed)
- **CUDA:** 11.2 or 11.8
- **cuDNN:** 8.9

**System Specifications:**
Intel i7 9th gen with 16GB RAM and GPU GTX1650 4GB VRAM.

**GPU Verification:**

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) #>>gave 1 denoting that GPU was found
print("TensorFlow built with CUDA: ", tf.test.is_built_with_cuda())  #>>gave True denoting Tensorflow build with CUDA support
```

**nvidia-smi (this command can be used in terminal cmd to check stats of gpu usage)**

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.set_visible_devices(gpus[0], 'GPU')  # Use the first GPU
except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
```

# Tested Object Detection Models:

1. **ssd_mobilenet_v2_coco_2018_03_29**
   - Code: *saved model format supporting code (2svedmodeltypeworkingimport cv2.py)*
   - Output: Displays a green box around the detected objects with a fixed label named "object".

2. **efficientdet_d0_coco17_tpu-32**
   - Code: *saved model format supporting code (attampt2.1.py)*
   - Output: Displays blue boxes around the identified objects, but occasionally mislabels them (e.g., cats as dogs, dogs as horses). A newer version of this code is available that uses a webcam instead of an ESP32 camera and provides more stable performance without sudden usage spikes.

3. **mobilenet_v2_1.4_224**
   - Code: *frozen.pb model type code (aug19.py)*
   - Output: Displays object classification on the top left with a matching range from >0 to 1, but no boxes are visible around the objects.

4. **ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8**
   - Code: *saved model format type and label map inside code (ESP_COCO.py)*
   - Output: Displays green boxes around identified objects with an accuracy above average. It can detect up to 90 objects, displaying a matching range of >0 to 1 for each object. The webcam version of this code has become more accurate than the ESP32 camera feed.

5. **efficientdet_d0_coco17_tpu-32**
   - Code: *same as model 2 with same code (webcam version)*
   - Output: Similar to model 2, but it is more laggy and has a higher load on the GPU. However, it performs significantly better than model 2 in identifying objects at times.

# Cloud TPU Testing:

The ESP versions of these codes have been tested successfully with Google's Cloud TPUs. More information will be added soon.
