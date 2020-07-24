import os
import pathlib
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PATH_TO_LABELS = 'mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_model(model_name):
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin= model_file,
    untar=True
  )
  model_dir = pathlib.Path(model_dir)/"saved_model"
  model = tf.saved_model.load(str(model_dir))
  return model

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  return output_dict

def show_inference(model, image_path):
  image_np = np.array(Image.open(image_path))
  output_dict = run_inference_for_single_image(model, image_np)
  
  regions = ""
  i = 0
  for c in output_dict['detection_classes']:
    if c == 1 and output_dict['detection_scores'][i] > 0.5:
      regions += " ".join(str(b) for b in output_dict['detection_boxes'][i]) + "\n"
    else:
      output_dict['detection_scores'][i] = float(0)
    i += 1

  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=None,
      use_normalized_coordinates=True,
      line_thickness=8
  )

  image_np_with_detections = Image.fromarray(image_np)
  image_np_with_detections.save(image_path[:-4] + "_detect.png")

  f = open(image_path[:-4] + "_detect.txt", "w")
  f.write(regions)
  f.close()

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

def human_detect(image_path):
  show_inference(detection_model, image_path)

import time
if __name__ == "__main__":
  start_time = time.time()
  # human_detect("trieu.jpg")
  # human_detect("images/sample.jpg")
  human_detect("../images/sample5.jpg")
  print("--- %s seconds ---" % (time.time() - start_time))

