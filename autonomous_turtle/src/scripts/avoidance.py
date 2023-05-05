#!/usr/bin/env python3

import signal
import sys
import threading
import cv2
import numpy as np
import pygame as pg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from pygame.locals import *
from sensor_msgs.msg import Image, LaserScan
from settings import *
from sprites import LDS, LocalRefFrame, Text


class AvoidanceRobot:
    def __init__(self):    
        # Initialize Pygame
        pg.init()
        pg.display.set_caption("Smart Autonomous Robot Laser Scan")
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.text = Text('Obstacle Avoidance Engaged', SCREEN_WIDTH/2, 30) 
        self.turn_left_text = Text('Turning Left', SCREEN_WIDTH/2, 80)
        self.turn_right_text = Text('Turning Right', SCREEN_WIDTH/2, 80)
        self.all_sprites = pg.sprite.Group()
        self.robot = LocalRefFrame((SCREEN_WIDTH/2), (SCREEN_HEIGHT/2), self.all_sprites)
        
        # State Variables for Obstacle Avoidance
        self.collision = False
        self.go_left = False
        self.go_right = False
        
        # Initialize LDS particles in coordinate system
        self.lds_particles = []
        for x in range(360):
            self.lds_particles.append(LDS(self.all_sprites, self.robot, x, 0))
            
        # Synchronization Primitives 
        self.scan_data_lock = threading.Condition()
        self.image_lock = threading.Lock()

        rospy.init_node("avoidance_robot", anonymous=True)
        rospy.loginfo('Initializing Avoidance Robot')
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.ros_lds_callback)
        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        self.ros_clock = rospy.Rate(10)

        # initialize twist message for velocity control
        self.tw = Twist()
        self.tw.linear.x = 0.0
        self.tw.angular.z = 0.0
        
        # initialize scan data
        self.ranges = np.zeros(360)

        # algorithm optimization (scan range)
        self.theta = int(np.rad2deg(np.arctan(ROBOT_RADIUS + ROBOT_BUFFER / COLLISION_DISTANCE)))
        
        # opencv bridge to convert ros image to opencv image
        self.bridge = CvBridge()

        # initialize threads for control, subscriber, camera, and pygame
        # set them as daemon to ensure they are killed when main thread exits
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.subscriber_thread = threading.Thread(target=self.subscriber_thread)
        self.subscriber_thread.daemon = True
        self.camera_thread = threading.Thread(target=self.image_sub_thread)
        self.camera_thread.daemon = True
        self.pygame_thread = threading.Thread(target=self.run)

        # Register signal handler to handle Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        sys.exit()
        
    def display_image(self, img):
        with self.image_lock:         
            cv2.imshow("Smart Autonomous Robot Camera Output", img)
            cv2.waitKey(3)
        
    def image_callback(self, img_msg):
        try:   
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError:
            rospy.logerr("CvBridge Error: {0}".format(CvBridgeError))
            
        self.display_image(cv_image)

    def ros_lds_callback(self, msg):
        with self.scan_data_lock:
            self.ranges = np.array(msg.ranges)
            self.ranges[np.isinf(self.ranges)] = 0
            self.scan_data_lock.notify_all()

    def control_loop(self):
        while not rospy.is_shutdown():
            decision = self.avoid_obstacle()
            self.tw.linear.x = decision[0]
            self.tw.angular.z = decision[1]
            self.vel_pub.publish(self.tw)
            rospy.loginfo("Sending Velocity Command: " + str(self.tw))
            self.ros_clock.sleep()

    def subscriber_thread(self):
        rospy.loginfo("Starting Laser Scan Subscriber")
        rospy.spin()
        
    def image_sub_thread(self):  
        rospy.loginfo("Starting Camera Subscriber")   
        rospy.spin()

    def avoid_obstacle(self):
        with self.scan_data_lock:
            self.scan_data_lock.wait()

            # check for collision using calculated range threshold
            for i in range(0, self.theta):
                if (self.ranges[i] <= COLLISION_DISTANCE and self.ranges[i] != 0) or (
                    self.ranges[360 - self.theta + i] <= COLLISION_DISTANCE
                    and self.ranges[360 - self.theta + i] != 0
                ):
                    self.collision = True
                    rospy.loginfo("Collision Detected. Generating Decision")
                    break
                else:
                    self.collision = False
                    
            # No collision detected, move forward
            if not self.collision:
                self.go_left = False
                self.go_right = False
                return [0.2, 0.0] 
            
            # Decision Already Made, Continue Turning
            if self.go_left:
                return [0.0, 0.2]
            elif self.go_right:
                return [0.0, -0.2]

            # make a decision by normalizing the ranges and computing weights
            normalized_set = self.ranges / np.max(self.ranges)

            # set the ranges to 1 or -1 where 0
            for i in range(0,180):
                if normalized_set[i] == 0:
                    normalized_set[i] = 1

            for i in range(180, 360):
                if normalized_set[i] == 0:
                    normalized_set[i] = -1
                else:
                    normalized_set[i] = -normalized_set[i]
                
            # get sum of the ranges
            if np.sum(normalized_set[270: 360]) + np.sum(normalized_set[0:90]) >= 0:
                rospy.loginfo('Decision: Turn Left')
                self.go_left = True
                return [0.0, 0.2]
            elif np.sum(normalized_set[270: 360]) + np.sum(normalized_set[0:90]) < 0:
                rospy.loginfo('Decision: Turn Right')
                self.go_right = True
                return [0.0, -0.2]

    def start(self):
        rospy.loginfo('Starting Obstacle Avoidance Robot Motion')
        self.running = True
        self.control_thread.start()
        self.subscriber_thread.start()
        self.camera_thread.start()
        self.pygame_thread.start()
        
    def run(self):
        while not rospy.is_shutdown():
            self.ros_clock.sleep()
            self.events()
            self.update()
            self.draw()
        
    def events(self):
        for event in pg.event.get():
            if event.type == QUIT:
                self.running = False
                pg.quit()
                cv2.destroyAllWindows()   
                sys.exit()
           
    def update(self):
        with self.scan_data_lock:
            self.scan_data_lock.wait()
            for x in range(360):
                self.lds_particles[x].refresh(x, self.ranges[x])    
          
        self.all_sprites.update()

    def draw(self):
        self.screen.fill(BLACK)
        self.all_sprites.draw(self.screen)
        if self.collision:
            self.screen.blit(self.text.image, self.text.rect)
        if self.go_left:
            self.screen.blit(self.turn_left_text.image, self.turn_left_text.rect)
        if self.go_right:
            self.screen.blit(self.turn_right_text.image, self.turn_right_text.rect)
        pg.display.flip()

if __name__ == "__main__":
    try: 
        avoidance_robot = AvoidanceRobot()
        avoidance_robot.start()
    except rospy.ROSInterruptException:
        pass