#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/impl/implicit_shape_model.hpp>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/visualization/cloud_viewer.h>
#include <std_srvs/Trigger.h>


class PcdConverter
{
public:
    PcdConverter()
    {
        sub = nh.subscribe("/bsd_sonar/pcl_preproc_sonar", 1, &PcdConverter::pc_callback, this);
        save_service = nh.advertiseService("avl/save_point_cloud", &PcdConverter::saveServiceCallback, this);
    }
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        ROS_INFO("Accumulating Data");
        
        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(*msg, cloud);

        accumulated_cloud += cloud;
        // viewer.showCloud(accumulated_cloud.makeShared());
    }
    void save_to_pcd()
    {
        if (!accumulated_cloud.empty())
        {
            pcl::io::savePCDFileASCII("point_cloud.pcd", accumulated_cloud);
            ROS_INFO("Saved Point PointCloud");
        }
        else
        {
            ROS_WARN("Accumulated PointCloud is empty. No data to save.");
        }
    }
    bool saveServiceCallback(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
    {
        save_to_pcd();
        res.success = true;
        res.message = "Saving accumulated point cloud triggered.";
        return true;
    }
    void spin()
    {
        ros::spin();
    }

private:
    pcl::PointCloud<pcl::PointXYZI> accumulated_cloud;
    // pcl::visualization::CloudViewer viewer;
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::ServiceServer save_service;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcd_converter_node");
    PcdConverter subscriber;
    subscriber.spin();
    subscriber.save_to_pcd();
    return 0;
}