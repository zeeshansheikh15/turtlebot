#!/usr/bin/env python

'''
Copyright (c) 2015, Mark Silliman
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# A very basic TurtleBot script that moves TurtleBot forward indefinitely. Press CTRL + C to stop.  To run:
# On TurtleBot:
# roslaunch turtlebot_bringup minimal.launch
# On work station:
# python goforward.py

from __future__ import print_function
from __future__ import division
import rospy
from geometry_msgs.msg import Twist
import cv2
from math import radians
from ar_markers import detect_markers
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
try:
    import cv2
except ImportError:
    raise Exception('Error: OpenCv is not installed')

from numpy import array, rot90

from ar_markers.coding import decode, extract_hamming_code
from ar_markers.marker import MARKER_SIZE, HammingMarker

BORDER_COORDINATES = [
    [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 0], [1, 6], [2, 0], [2, 6], [3, 0],
    [3, 6], [4, 0], [4, 6], [5, 0], [5, 6], [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6],
]

ORIENTATION_MARKER_COORDINATES = [[1, 1], [1, 5], [5, 1], [5, 5]]

marker_id  = 0



def validate_and_turn(marker):
    # first, lets make sure that the border contains only zeros
    for crd in BORDER_COORDINATES:
        if marker[crd[0], crd[1]] != 0.0:
            raise ValueError('Border contians not entirely black parts.')
    # search for the corner marker for orientation and make sure, there is only 1
    orientation_marker = None
    for crd in ORIENTATION_MARKER_COORDINATES:
        marker_found = False
        if marker[crd[0], crd[1]] == 1.0:
            marker_found = True
        if marker_found and orientation_marker:
            raise ValueError('More than 1 orientation_marker found.')
        elif marker_found:
            orientation_marker = crd
    if not orientation_marker:
        raise ValueError('No orientation marker found.')
    rotation = 0
    if orientation_marker == [1, 5]:
        rotation = 1
    elif orientation_marker == [5, 5]:
        rotation = 2
    elif orientation_marker == [5, 1]:
        rotation = 3
    marker = rot90(marker, k=rotation)
    return marker


def detect_markers(img):
    """
    This is the main function for detecting markers in an image.

    Input:
      img: a color or grayscale image that may or may not contain a marker.

    Output:
      a list of found markers. If no markers are found, then it is an empty list.
    """
    if len(img.shape) > 2:
        width, height, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        width, height = img.shape
        gray = img

    edges = cv2.Canny(gray, 10, 100)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]

    # We only keep the long enough contours
    min_contour_length = min(width, height) / 50
    contours = [contour for contour in contours if len(contour) > min_contour_length]
    warped_size = 49
    canonical_marker_coords = array(
        (
            (0, 0),
            (warped_size - 1, 0),
            (warped_size - 1, warped_size - 1),
            (0, warped_size - 1)
        ),
        dtype='float32')

    markers_list = []
    for contour in contours:
        approx_curve = cv2.approxPolyDP(contour, len(contour) * 0.01, True)
        if not (len(approx_curve) == 4 and cv2.isContourConvex(approx_curve)):
            continue

        sorted_curve = array(
            cv2.convexHull(approx_curve, clockwise=False),
            dtype='float32'
        )
        persp_transf = cv2.getPerspectiveTransform(sorted_curve, canonical_marker_coords)
        warped_img = cv2.warpPerspective(img, persp_transf, (warped_size, warped_size))

        # do i really need to convert twice?
        if len(warped_img.shape) > 2:
            warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        else:
            warped_gray = warped_img

        _, warped_bin = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
        marker = warped_bin.reshape(
            [MARKER_SIZE, warped_size // MARKER_SIZE, MARKER_SIZE, warped_size // MARKER_SIZE]
        )
        marker = marker.mean(axis=3).mean(axis=1)
        marker[marker < 127] = 0
        marker[marker >= 127] = 1

	global marker_id
        try:
            marker = validate_and_turn(marker)
            hamming_code = extract_hamming_code(marker)
            marker_id = int(decode(hamming_code), 2)
            markers_list.append(HammingMarker(id=marker_id, contours=approx_curve))
        except ValueError:
            continue
    return markers_list


class GoForward():
    def __init__(self):
        # initiliaze
        rospy.init_node('GoForward', anonymous=False)

	# tell user how to stop TurtleBot
	rospy.loginfo("To stop TurtleBot q")

        # What function to call when you ctrl + c    
        rospy.on_shutdown(self.shutdown)
        
	# Create a publisher which can "talk" to TurtleBot and tell it to move
        # Tip: You may need to change cmd_vel_mux/input/navi to /cmd_vel if you're not using TurtleBot2
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)
     
	#TurtleBot will stop if we don't keep telling it to move.  How often should we tell it to move? 10 HZ
        r = rospy.Rate(10);

        # Twist is a datatype for velocity
        move_cmd = Twist()
	# let's go forward at 0.2 m/s
        move_cmd.linear.x = 0.2
	# let's turn at 0 radians/s
	#move_cmd.angular.z = 0

        #let's turn at 45 deg/s
        turn_cmd = Twist()
        turn_cmd.linear.x = 0
        turn_cmd.angular.z = radians(50); #45 deg/s in radians/s



	#Capture the workstation video input
	capture = cv2.VideoCapture(0)
	
        if capture.isOpened():  # try to get the first frame
                frame_captured, frame = capture.read()
        else:
                frame_captured = False
	
	global i
	i = 0
	while frame_captured:

		print("---------------------")
                markers = detect_markers(frame)
		if detect_markers(frame):  # try to get the first frame
                	#print("detected")			
			
			for marker in markers:
				print ("111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
				marker.highlite_marker(frame)
				print (marker_id)
				if (marker_id == 1803):
					if (i == 1):
						print("break out")
						continue
					print("***************************up  right**********************************")
					i = 1
					for x in range(0,90):
				    		self.cmd_vel.publish(move_cmd)
						print("moving forward")
				    		# wait for 0.1 seconds (10 HZ) and publish again
				    		r.sleep()
					for x in range(0,60):
						self.cmd_vel.publish(turn_cmd)
						print("ROUND MOVEEE")
						r.sleep() 
					for x in range(0,70):
				    		self.cmd_vel.publish(move_cmd)
						print("moving forward")
				    		# wait for 0.1 seconds (10 HZ) and publish again
				    		r.sleep()
		
				elif(marker_id == 123):
					print("***************************round and up***************************************")
					if (i == 2):
						print("break out")
						continue
					i = 2
					for x in range(0,35):
						self.cmd_vel.publish(turn_cmd)
						print("ROUND MOVEEE")
						r.sleep() 
					for x in range(0,120):
				    		self.cmd_vel.publish(move_cmd)
						print("moving forward")
				    		# wait for 0.1 seconds (10 HZ) and publish again
				    		r.sleep()


                cv2.imshow('Test Frame', frame)
		
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                frame_captured, frame = capture.read()

	



        # When everything done, release the capture
        capture.release()
        cv2.destroyAllWindows()
	self.shutdown()

	# as long as you haven't ctrl + c keeping doing...
       # while not rospy.is_shutdown():
	    # publish the velocity
        #    self.cmd_vel.publish(move_cmd)
	    # wait for 0.1 seconds (10 HZ) and publish again
         #   r.sleep()
                        
        
    def shutdown(self):
        # stop turtlebot
        rospy.loginfo("Stop TurtleBot")
	# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
        self.cmd_vel.publish(Twist())
	# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
        rospy.sleep(1)
 
if __name__ == '__main__':
    try:
        GoForward()
    except:
        rospy.loginfo("GoForward node terminated.")

