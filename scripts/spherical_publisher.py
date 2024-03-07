#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

def generate_sphere_points(radius, num_points):
    points = []
    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points.append([x, y, z])
    return points

def create_point_cloud2_msg(points):
    header = rospy.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'world'
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]
    return pc2.create_cloud(header, fields, points)

def point_cloud_publisher():
    rospy.init_node('point_cloud_publisher', anonymous=True)
    pub = rospy.Publisher('/bsd_sonar/pcl_preproc_sonar', PointCloud2, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Generate all sphere points once, outside the loop
    sphere_points = generate_sphere_points(20, 10000)  # 10 meters radius, 100 points
    cloud_msg = create_point_cloud2_msg(sphere_points)

    while not rospy.is_shutdown():
        # Publish the complete set of sphere points in each iteration
        pub.publish(cloud_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        point_cloud_publisher()
    except rospy.ROSInterruptException:
        pass
