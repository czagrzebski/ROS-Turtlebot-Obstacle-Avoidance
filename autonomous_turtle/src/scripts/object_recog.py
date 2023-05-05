#!/usr/bin/env python3

import signal
import sys
import threading
import cv2 
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np


class AvoidanceRobot:
    def __init__(self):          
        rospy.init_node("avoidance_robot", anonymous=True)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.image_lock = threading.Lock()
        self.bridge = CvBridge()
        
        # Load Yolo
        print("LOADING YOLO")
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #save all the names in file o the list classes
        self.classes = []
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        #get layers of the network
        layer_names = self.net.getLayerNames()
        #Determine the output layer names from the YOLO model 
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        print("YOLO LOADED")

        self.camera_thread = threading.Thread(target=self.image_sub_thread)

    def signal_handler(self, sig, frame):
        sys.exit()
        
    def display_image(self, img):
        # Using blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
        #Detecting objects
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        height, width, channels = img.shape

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        #We use NMS function in opencv to perform Non-maximum Suppression
        #we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
                1/2, color, 2)
        cv2.imshow("Smart Autonomous Robot Camera Output", img)
        cv2.waitKey(3)
        
    def image_callback(self, img_msg):
        try:   
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError:
            rospy.logerr("CvBridge Error: {0}".format(CvBridgeError))
            
        self.display_image(cv_image)

    def image_sub_thread(self):     
        rospy.spin()

    def start(self):
        self.running = True
        self.camera_thread.start()

if __name__ == "__main__":
    try: 
        avoidance_robot = AvoidanceRobot()
        avoidance_robot.start()
    except rospy.ROSInterruptException:
        pass