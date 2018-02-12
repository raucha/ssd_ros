#!/usr/bin/env python
# coding: utf-8

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility

import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
# from geometry_msgs.msg import Twist
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# %matplotlib inline
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

input_shape=(300, 300, 3)
model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights('/home/akio-ubuntu/catkin_ws/src/pythontest/scripts/weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

# 事前に適当に呼び出しておく
inputs = []
# img_path = '/home/akio-ubuntu/github/ssd_keras/pics/fish-bike.jpg'
# img = image.load_img(img_path, target_size=(300, 300))
# img = image.img_to_array(img)
img = np.zeros((300,300,3))
inputs.append(img)
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
pub_array = None
pub_image = None
LATEST_IMAGE = None


def callback(data):
    global LATEST_IMAGE
    LATEST_IMAGE = data
    return


def main():
    rospy.init_node('ros_ssd', anonymous=False)
    global pub_array, pub_image
    pub_array = rospy.Publisher('/class_num', String, queue_size=10)
    pub_image = rospy.Publisher('/classed_image', Image, queue_size=10)
    rospy.Subscriber("/usb_cam/image_raw", Image, callback, queue_size=1)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        if LATEST_IMAGE is not None:
            get_class(LATEST_IMAGE)
        rate.sleep()


def get_class(arg):
    ## 認識用の画像の準備
    try:
        cv_image = CvBridge().imgmsg_to_cv2(arg, "rgb8")
        # cv_image = CvBridge().imgmsg_to_cv2(arg, "bgr8")
    except CvBridgeError as e:
        print(e)
    # (rows,cols,channels) = cv_image.shape
    # rospy.loginfo("got image")
    # 入力画像の設定
    inputs = []
    # img_path = './pics/boys.jpg'
    # img = image.load_img(img_path, target_size=(300, 300))
    # img = image.img_to_array(img)
    # inputs.append(img.copy())
    img = cv2.resize(cv_image, (300, 300))
    # print cv_image.shape
    inputs.append(img)
    inputs = preprocess_input(
        np.array(inputs, dtype="float32"))
    # print inputs

    # クラス分け実行
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    # # %%time
    # a = model.predict(inputs, batch_size=1)
    # b = bbox_util.detection_out(preds)

    # Parse the outputs.
    # 画像は1枚なので全て[0]の結果を利用する
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]


    # キャンバス生成
    fig = plt.figure()
    plt.imshow(cv_image / 255.)
    currentAxis = fig.gca()
    # currentAxis = plt.gca()
    # クラスごとの色生成
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()

    # 検出された各物体に対してループ
    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * cv_image.shape[1]))
        ymin = int(round(top_ymin[i] * cv_image.shape[0]))
        xmax = int(round(top_xmax[i] * cv_image.shape[1]))
        ymax = int(round(top_ymax[i] * cv_image.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label_name)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
        rospy.loginfo("{} {} {} {}".format(xmin, ymin, xmax, ymax))
        # plt.show()
        # plt.pause(0.1)
        res = String()
        res.data = "{} {} {} {} {}".format(label_name, coords[0][0], coords[0][1], coords[1], coords[2])
        pub_array.publish(res)
    pub_image.publish(CvBridge().cv2_to_imgmsg(pltfig2cvimage(fig), "bgr8"))
    return


def pltfig2cvimage(fig):
    fig.canvas.draw()
    cvimg = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8, sep="")
    cvimg = cvimg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cvimg = cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR)
    return cvimg


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass