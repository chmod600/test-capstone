# Credits for traffic light detector - Eric Tang of our group BlueBirdDynamics
# We tried two different approaches, object detection API and using code
# from the traffic light detector.
# I was trying to get the object detection API work, but it was too slow on our environment
# Eric compiled the code from our traffic sign detector project and using transfer learning
# we are able to somewhat detect traffic lights.
# The traffic light detector needs more training to be accurate, so we are now putting that
# on hold and trying to tune the car to go 50 mph.

#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from light_classification.tl_classifier import TLClassifier
import tf
import tensorflow as tf2
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
import yaml
from scipy.spatial import KDTree
from cv_bridge import CvBridge, CvBridgeError

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        self.waypoints_2d = None
        self.bridge = CvBridge()

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        # We might want to use image_raw here
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [
                [
                    waypoint.pose.pose.position.x,
                    waypoint.pose.pose.position.y
                ] for waypoint in waypoints.waypoints
            ]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        light_wp, state = self.process_traffic_lights(msg)

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light, msg):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        TL_STATE = 4
        ypoint=100
        xpoint=250
        height=352
        width=352
        isTraining = False
        if(isTraining == True):
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cropping123 = cv_image[ypoint:ypoint+height, xpoint:xpoint+width, :]
            cv_image = resize(cv_image, (32, 32))
            cv2.imwrite("/home/student/images2/" + str(msg.header.seq) + ".png", cropping123)
        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            cv_image = cropping123 = cv_image[ypoint:ypoint+height, xpoint:xpoint+width, :]
            cv_image = resize(cv_image, (32, 32))
            cv_image = (cv_image-0.5)/0.5
            with tf2.Session(graph=tf2.Graph()) as sess:
                tf2.saved_model.loader.load(sess, ["serve"], "./outModel")
                graph = tf2.get_default_graph()
                input1 = graph.get_tensor_by_name("x:0")
                output = graph.get_tensor_by_name("add_2:0")
                prediction = sess.run(tf2.argmax(output, 1), {input1: [cv_image]})
                pred1 = prediction[0]
                if pred1 == 1:
                    TL_STATE = 0
                elif pred1 == 2:
                    TL_STATE = 2
                fname = str(prediction) + "_" + str(msg.header.seq)
                bgrCropped = cv2.cvtColor(cropping123, cv2.COLOR_RGB2BGR)
                cv2.imwrite("/home/student/images2/" + fname + ".png", bgrCropped)

        return TL_STATE

    def process_traffic_lights(self, msg):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            diff = len(self.waypoints.waypoints)
            #rospy.loginfo("--->begin")
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])

                d = temp_wp_idx - car_wp_idx
                #rospy.loginfo("this is diff %f", d)
                if d >= 0 and d < diff and d < 50:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx
            #rospy.loginfo("--->end")

        if closest_light:
            state = self.get_light_state(closest_light, msg)
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
