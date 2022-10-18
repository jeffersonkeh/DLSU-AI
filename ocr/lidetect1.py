import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from collections import defaultdict
from io import StringIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import cv2

from api import label_map_util
from api import visualization_utils as vis_util

PATH_TO_CKPT = 'api/frozen_inference_graph0.pb'
PATH_TO_LABELS = 'api/object-detection.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

video = cv2.VideoCapture('ptuazon3.avi')
out = cv2.VideoWriter('platedetect.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,(1280,480))
while(video.isOpened()):
    black1 = np.zeros((240,640,3),np.uint8)
    ret, frame = video.read()
    height = 480
    width  = 640
    if ret == True:
        frame = cv2.resize(frame,(640,480))
        origframe = frame.copy()
        frame_expanded = np.expand_dims(frame, axis=0)
    else:
        break
    
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    image, xmin, xmax, ymin, ymax = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        max_boxes_to_draw=3,
        min_score_thresh=0.90)
    ymin = int(ymin * height)
    xmin = int(xmin * width)
    ymax = int(ymax * height)
    xmax = int(xmax * width)
    dims = [xmin, xmax, ymin, ymax]
    checkZero = any(n<=0 for n in dims)
    if (checkZero == False) and (ymin > 240):
        plateImg = origframe[ymin:ymax,xmin:xmax]
        imgcrop = cv2.resize(plateImg,(560,200))
        black1[20:220,40:600] = imgcrop

    plateArea = image[240:480,0:640]
    vstack1 = np.vstack((black1,plateArea))
    vstack1 = cv2.resize(vstack1,(640,480))
    hstack1 = np.hstack((image,vstack1))
    cv2.line(hstack1,(641,0),(641,480),(255,255,255),3)
    cv2.imshow('NSTW - Plate Detection',hstack1)
    out.write(hstack1)
    k = cv2.waitKey(30)
    if k==27:    # Esc key to stop
      break
    
cv2.destroyAllWindows()
video.release()
