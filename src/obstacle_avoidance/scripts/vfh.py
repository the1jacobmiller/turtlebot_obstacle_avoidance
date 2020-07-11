#!/usr/bin/python
import rospy
import cv2
import math
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from cv_bridge import CvBridge, CvBridgeError

# height = 480, width = 640
columns_per_bucket = 128
num_buckets = 640 // 128
obstacle_threshold = 1000.
angle_threshold = math.pi/6.
max_height = 200

current_location = (0.,0.)
current_yaw = 0.
prev_bucket = 0
prev_rotate_factor = 0.
prev_linear_x = 0
count = 0

target_pose = (0.0, 3.0)

pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)

cv2.namedWindow('Test')
cv2.resizeWindow('Test', 300, 300)

def quaternion_to_RPY(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll,pitch,yaw

def calcAngle2Point():
    diff_x = target_pose[0]-current_location[0]
    if diff_x == 0.:
        diff_x = 0.00001
    diff_y = target_pose[1]-current_location[1]
    angle = math.atan(diff_y/diff_x)
    print('current_location: ', current_location)
    print('target_pose: ', target_pose)
    if (diff_x < 0. and diff_y > 0.):
        angle = math.pi + angle
    elif (diff_x < 0. and diff_y < 0.):
        angle = -math.pi + angle
    return angle

def calcAngleDiff(current_yaw, angle_to_target):
    angle_diff = current_yaw - angle_to_target
    if angle_diff > math.pi:
        angle_diff = angle_diff - 2.*math.pi
    elif angle_diff < -math.pi:
        angle_diff = angle_diff + 2.*math.pi
    return angle_diff

def createPolarHistogram(cv_image):
    # returns a polar histogram list describing obstacle density
    h = cv_image.shape[0]
    w = cv_image.shape[1]

    num_buckets = w // columns_per_bucket
    bucket_total = [0] * num_buckets
    polar_histogram = [0] * num_buckets

    # loop through pixels in cv_image
    for y in range(max_height, h):
        for x in range(0, w):
            bucket = x // columns_per_bucket
            pixel_intensity = cv_image[y,x]
            bucket_total[bucket] += pixel_intensity

    # populate polar histogram with average pixel intensity values
    for i in range(0, len(bucket_total)):
        polar_histogram[i] = bucket_total[i] / ((h-max_height)*columns_per_bucket)

    return polar_histogram

def findSafeBuckets(polar_histogram):
    safe_buckets = []
    for i in range(0, len(polar_histogram)):
        if polar_histogram[i] > obstacle_threshold:
            safe_buckets.append(i)
    return safe_buckets

def maskSafeBuckets(safe_buckets, cv_image):
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    h = cv_image.shape[0]
    w = cv_image.shape[1]
    # loop through pixels in cv_image
    for y in range(max_height, h):
        for x in range(0, w):
            bucket = x // columns_per_bucket
            pixel = rgb_image[y,x]
            # print(pixel)
            if bucket in safe_buckets:
                rgb_image[y,x,1] += 7000

    cv2.imshow('Test', rgb_image)
    cv2.waitKey(0)
    print("Finished masking image")

def calcControl(safe_buckets):
    global prev_rotate_factor
    global prev_linear_x
    global count
    max_count = 8
    middle_bucket = num_buckets // 2
    angle_to_target = calcAngle2Point()
    angle_diff = calcAngleDiff(current_yaw, angle_to_target)
    print('Angle to target: ', angle_to_target)
    print('Current yaw', current_yaw)
    print('Angle Diff: ', angle_diff)

    delta = ((current_location[0] - target_pose[0])**2 + (current_location[1] - target_pose[1])**2)**.5
    if delta <= 0.5:
        print("Success! Distance to Target: ", delta)
        exit()

    if angle_diff > angle_threshold:
        target_bucket = middle_bucket+1
    elif -angle_diff > angle_threshold:
        target_bucket = middle_bucket-1
    else:
        target_bucket = middle_bucket

    if prev_linear_x != 0. and count < max_count:
        target_bucket = middle_bucket
        count += 1
    else:
        count = 0

    linear_x = 0.
    rotate_factor = 0.
    if target_bucket in safe_buckets: # preferred trajectory is safe
        if target_bucket == middle_bucket: # go straight
            linear_x = 0.05
        else: # turn towards target
            rotate_factor = middle_bucket - target_bucket
    else: # preferred trajectory is blocked - choose next best option
        closest_bucket = -1
        for bucket in safe_buckets:
            if closest_bucket == -1 or abs(target_bucket - bucket) < abs(target_bucket - closest_bucket):
                closest_bucket = bucket
                rotate_factor = middle_bucket - bucket
        if closest_bucket == middle_bucket:
            linear_x = 0.05
    if prev_rotate_factor != 0 and linear_x == 0.:
        rotate_factor = prev_rotate_factor
    if rotate_factor == 0. and linear_x == 0.:
        rotate_factor = 0.1
    prev_rotate_factor = rotate_factor
    angular_z = rotate_factor * 0.1

    prev_linear_x = linear_x
    return linear_x, angular_z

def publishControl(linear_x, angular_z):
    # publishes control command to the TurtleBot
    cmd = Twist()
    cmd.linear.x = linear_x
    cmd.angular.z = angular_z
    print(cmd)
    pub.publish(cmd)

def imageCallback(msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    polar_histogram = createPolarHistogram(cv_image)
    safe_buckets = findSafeBuckets(polar_histogram)
    print(safe_buckets)
    # maskSafeBuckets(safe_buckets, cv_image)
    linear_x, angular_z = calcControl(safe_buckets)
    publishControl(linear_x, angular_z)

def localizationCallback(msg):
    global current_location, current_yaw

    msg_pose = msg.pose.pose
    current_location = (msg_pose.position.x, msg_pose.position.y)
    roll,pitch,current_yaw = quaternion_to_RPY(msg_pose.orientation.x, msg_pose.orientation.y, msg_pose.orientation.z, msg_pose.orientation.w)

if __name__ == '__main__':
    rospy.init_node('vfh_node', anonymous=True)
    rate = rospy.Rate(10) # 10hz

    rospy.Subscriber("/camera/depth/image_raw", Image, imageCallback, queue_size = 1)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, localizationCallback, queue_size = 1)

    rospy.spin()
