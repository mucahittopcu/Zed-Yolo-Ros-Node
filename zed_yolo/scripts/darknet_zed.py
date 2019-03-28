#!/usr/bin/env python
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Original *nix 2.7: https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version: https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn, Aymeric Dujardin
@date: 20180911
"""
# pylint: disable=R, W0401, W0614, W0703
import cv2
#import pyzed.sl as sl
from ctypes import *
import math
import random
import os
import numpy as np
import statistics
import sys
import getopt
from random import randint

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage, PointCloud2

#np.set_printoptions(threshold=np.nan)

netMain = None
metaMain = None
altNames = None

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1


def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("/home/nvidia/libdarknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("darknet.so", RTLD_GLOBAL)
#lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)
hasGPU = True

lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

if hasGPU:
    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
    c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        if altNames is None:
            nameTag = meta.names[i]
        else:
            nameTag = altNames[i]
        res.append((nameTag, out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
    """
    Performs the detection
    """
    custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    custom_image = cv2.resize(custom_image, (lib.network_width(
        net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
    im, arr = array_to_image(custom_image)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]
    if nms:
        do_nms_sort(dets, num, meta.classes, nms)
    res = []
    if debug:
        print("about to range")
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                if altNames is None:
                    nameTag = meta.names[i]
                else:
                    nameTag = altNames[i]
                res.append((nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
    res = sorted(res, key=lambda x: -x[1])
    free_detections(dets, num)
    return res

def getObjectDepth(depth, bounds):
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x = statistics.median(x_vect)
        y = statistics.median(y_vect)
        z = statistics.median(z_vect)
    except Exception:
        x = -1
        y = -1
        z = -1
        pass

    return x, y, z


def generateColor(metaPath):
    random.seed(42)
    f = open(metaPath, 'r')
    content = f.readlines()
    class_num = int(content[0].split("=")[1])
    color_array = []
    for x in range(0, class_num):
        color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    return color_array

class BaseClass(object):
    def __init__(self):
        rospy.init_node("zed_yolo")

        self.debug = False
        self.rate = rospy.Rate(30)
        self.image = None
        self.depth = None
        self.image_ready = False
        self.depth_ready = False
        self.thresh = 0.25
        self.darknet_path="/home/nvidia/libdarknet/"
        self.configPath = self.darknet_path + "cfg/traffic_signs.cfg"
        self.weightPath = "/home/nvidia/racecar-openzeka/src/FourPlusOne/zed_yolo/scripts/traffic_signs_2000.weights"
        self.metaPath = "/home/nvidia/racecar-openzeka/src/FourPlusOne/zed_yolo/scripts/traffic_signs.data"

        # Import the global variables. This lets us instance Darknet once, then just call performDetect() again without instancing again
        global metaMain, netMain, altNames  # pylint: disable=W0603

        self.pub = rospy.Publisher("zed_yolo_detect",String,queue_size=15)
        #rospy.Subscriber('/left/image_rect_color', Image, self.zed_image_callback, queue_size=1)
        rospy.Subscriber('/zed/left/image_rect_color/compressed', CompressedImage, self.zed_image_callback, queue_size=1)
        #rospy.Subscriber('/depth/depth_registered', Image, zed_depth_callback, queue_size=1)
        rospy.Subscriber('/zed/point_cloud/cloud_registered', PointCloud2, self.zed_depth_callback, queue_size=1)

        assert 0 < self.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                            os.path.abspath(self.metaPath)+"`")
        if netMain is None:
            netMain = load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if metaMain is None:
            metaMain = load_meta(self.metaPath.encode("ascii"))
        if altNames is None:
            # In thon 3, the metafile default access craps out on Windows (but not Linux)
            # Read the names file and create a list to feed to detect
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                    re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        self.color_array = generateColor(self.metaPath)

        print("Running...")


    def zed_image_callback(self, data):
            #print("### image callback ###")
            np_arr = np.fromstring(data.data, np.uint8)
            #if CompressedImage use cv2.imdecode
            self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if not self.image_ready:
                self.image_ready = True

            if self.debug:
                cv2.imshow('Image', self.image)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

    def zed_depth_callback(self, data):
            #print("### depth callback ###")
            np_arr = np.fromstring(data.data, np.float32)
            self.depth = np_arr.reshape(720, 1280, 4)
            if not self.depth_ready:
                self.depth_ready = True
        
        
    def pipeline(self):
            try:
                if self.image_ready and self.depth_ready:
                    img = self.image
                    detections = detect(netMain, metaMain, img, self.thresh)

                    #chr(27) + "[2J"+
                    print("**** " +
                        str(len(detections)) + " Results ****")
                    for detection in detections:
                        label = detection[0]
                        confidence = detection[1]
                        pstring = label+": "+str(np.rint(100 * confidence))+"%"
                        print(pstring)
                        bounds = detection[2]
                        yExtent = int(bounds[3])
                        xEntent = int(bounds[2])
                        # Coordinates are around the center
                        xCoord = int(bounds[0] - bounds[2]/2)
                        yCoord = int(bounds[1] - bounds[3]/2)
                        boundingBox = [ [xCoord, yCoord], [xCoord, yCoord + yExtent], [xCoord + xEntent, yCoord + yExtent], [xCoord + xEntent, yCoord] ]
                        thickness = 1
                        x, y, z = getObjectDepth(self.depth, bounds)
                        distance = math.sqrt(x * x + y * y + z * z)
                        distance = "{:.2f}".format(distance)
                        rospy.loginfo("Label: " + pstring + " # Distance: " + str(distance))
                        pubSendInfo="Label:"+label+";Confidence"+str(np.rint(100 * confidence))+";Distance:"+str(distance)
                        #cv2.rectangle(img, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord+(18 +thickness*4)), self.color_array[detection[3]], -1)
                        #cv2.putText(img, label + " " +  (str(distance) + " m"), (xCoord+(thickness*4), yCoord+(10 +thickness*4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        #cv2.rectangle(img, (xCoord-thickness, yCoord-thickness), (xCoord + xEntent+thickness, yCoord + yExtent+thickness), self.color_array[detection[3]], int(thickness*2))
                        self.pub.publish(pubSendInfo)
                    #cv2.imshow("ZED", img)
                    #key = cv2.waitKey(5)
                else:
                    pass #key = cv2.waitKey(5)

            except Exception as e:
                print(e)
            #cv2.destroyAllWindows()
            self.rate.sleep()

        
if __name__ == "__main__":
    zed = BaseClass()
    while not rospy.is_shutdown():
        zed.pipeline()
    rospy.spin()
