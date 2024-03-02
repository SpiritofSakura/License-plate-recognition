#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import requests
import json
import urllib.request
from difflib import SequenceMatcher  # 6 sekund
import distance  # 8 sekund
import jellyfish
import string
import re
from tkinter import *
import SendPush as send


# switch funkcija

def switch(case):
    sporocilo = {
        "parkiral": "Vaše vozilo je parkirano.",
        "zasedeno": "Parkiriščje je polno! Tukaj trenutno ne morete parkirati.",
        "odsel": "Zapustili ste parkirišče."

    }
    return sporocilo.get(case, "Neveljavno")


# In[ ]:

CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

# In[ ]:


paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH': os.path.join('Tensorflow', 'protoc')
}

# In[ ]:


files = {
    'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# In[ ]:


for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            try:
                os.makedirs(path)
            except:
                print('napaka')
            # get_ipython().system('mkdir -p {path}')

        if os.name == 'nt':
            try:
                os.makedirs(path)
            except:
                print('napaka')
            # get_ipython().system('mkdir {path}')

# In[ ]:


labels = [{'name': 'licence', 'id': 1}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# In[ ]:


import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# In[ ]:


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-21')).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


# In[ ]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# In[ ]:


IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', 'Cars417.png')

# In[ ]:


img = cv2.imread(IMAGE_PATH)
image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=5,
    min_score_thresh=.8,
    agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()

# In[ ]:


# In[ ]:


detection_threshold = 0.7

# In[ ]:


image = image_np_with_detections
scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

# In[ ]:


width = image.shape[1]
height = image.shape[0]

# In[ ]:


import easyocr

# In[ ]:


detection_threshold = 0.7

# In[ ]:


image = image_np_with_detections
scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
boxes = detections['detection_boxes'][:len(scores)]
classes = detections['detection_classes'][:len(scores)]

# In[ ]:


width = image.shape[1]
height = image.shape[0]

# In[ ]:


region_threshold = 0.05


# In[ ]:


def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]

    plate = []
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))

        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate


# In[ ]:


region_threshold = 0.05


# In[ ]:


def ocr_it(image, detections, detection_threshold, region_threshold):
    # Scores, boxes and classes above threhold
    scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    classes = detections['detection_classes'][:len(scores)]

    # Full image dimensions
    width = image.shape[1]
    height = image.shape[0]

    # Apply ROI filtering and OCR
    for idx, box in enumerate(boxes):
        roi = box * [height, width, height, width]
        region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)

        text = filter_text(region, ocr_result, region_threshold)

        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.show()
        print(text)
        return text, region


# In[ ]:


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.5,
        agnostic_mode=False)

    booltest = False

    try:

        text, region = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)
        spremenljivka = ' '.join([str(elem) for elem in text])

        # spremenljivka = "'" + spremenljivka + "'"
        # req = urllib.request.urlopen(url=f'https://www.spiritofsakura.eu/seminarska/test.php?reg_st={spremenljivka}')
        # print(req.read())
        pattern = '[^A-Z0-9]+'
        y = re.sub(pattern, ' ', spremenljivka)
        ret1 = y.split()  # PODATEK RAZDELI NA VEC DELOV

        booltest = True
        response_API = requests.get('https://spiritofsakura.eu/seminarska/primerjaj.php%27')
        # print("a")
        data = response_API.text
        parse_json = json.loads(data)

        for x in range(len(parse_json)):
            active_case = parse_json[x]

            token = active_case['token']
            # for item in active_case:

            string1 = active_case['reg_st']
            string = string1.replace("-", " ")  # PREVERI KAJ STA PRVI 2 crki IZ BAZE (LJ,MB, CE,SG, NG, itd...)
            ret2 = string.split()
            if (ret1[0] == ret2[0]):
                # s = SequenceMatcher(None,y,string)
                s = jellyfish.jaro_similarity(y, string)

                # natančnost = distance.levenshtein(y,string)
                print(s)
                if (s >= 0.90):
                    print(y, " | ", string, " = ", s, "SE UJEMA")

                    # -------------PODATEK zapiše v bazo-------------------
                    # zapis = "'" +string1+"'"
                    # print(zapis)
                    # url=f'https://www.spiritofsakura.eu/seminarska/test.php?reg_st={zapis}'
                    # req = urllib.request.urlopen(url=url)
                    # print(req)
                    url = 'https://www.spiritofsakura.eu/seminarska/test.php'
                    myobj = {'registrska': string1}  # string1 je tisto kar je pravilno iz baze;

                    x = requests.post(url, data=myobj)

                    # pridobimo kaj vrne funkcija
                    akcija = x.text.strip()
                    akcija = akcija.split(',')

                    send.sendPush("Obvestilo", switch(akcija[0]), token, akcija)

                    # cap.release()
                    # cv2.destoryAllWindows()
                    break
                else:
                    print(y, " | ", string, " = ", s, "SE NE UJEMA")






    except Exception as e:
        if booltest == True:
            print(e)
            booltest = False
        pass
