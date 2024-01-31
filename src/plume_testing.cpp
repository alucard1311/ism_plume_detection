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
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <std_srvs/Trigger.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/pfh.h>
#include <mutex>
#include <thread>
#include <chrono>

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud with Normals"));

class PcdConverter
{
public:
    PcdConverter() : counter(0)
    {
        sub = nh.subscribe("/bsd_sonar/pcl_preproc_sonar", 1, &PcdConverter::pc_callback, this);
        save_service = nh.advertiseService("avl/save_point_cloud", &PcdConverter::saveServiceCallback, this);
        viewer_timer = nh.createTimer(ros::Duration(1), &PcdConverter::timerCB, this);
    }
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {

        ROS_INFO("Processing Data");

        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        // voxel_grid.setInputCloud(cloud);
        // voxel_grid.setLeafSize(0.4, 0.4, 0.01); // Adjust leaf size as needed
        // voxel_grid.filter(*cloud);

        // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        // sor.setInputCloud(cloud);
        // sor.setMeanK(50);            // Adjust as needed
        // sor.setStddevMulThresh(1.0); // Adjust as needed
        // sor.filter(*cloud);

        // accumulated_cloud += cloud;
        //  viewer.showCloud(accumulated_cloud.makeShared());

        // Surface Normal Estimation
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud);
        normal_estimator.setRadiusSearch(10);
        normal_estimator.compute(*cloud_normals);
        ROS_INFO("Done Computing Normals Data");
        std::string cloud_name = "cloud" + std::to_string(counter);
        viewer->addPointCloud(cloud, cloud_name);
        std::string normals_name = "normals" + std::to_string(counter);

        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 10, 0.02, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 30, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0, 0.0, 0.0, normals_name);
        // fpfh estimation
        // pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
        // pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
        // fpfh_estimation.setInputCloud(cloud);           // Assuming you're working with a single cloud
        // fpfh_estimation.setInputNormals(cloud_normals); // Assuming you've computed normals
        // fpfh_estimation.setRadiusSearch(40);            // Adjust the radius based on your data
        // fpfh_estimation.compute(*fpfh_features);

        //std::thread pfh_thread(&PcdConverter::computePFH, this, cloud, cloud_normals);
        //pfh_thread.join();

        ROS_INFO("Im at the end");
        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 255, 0, 0);
        // viewer->addPointCloud<pcl::PointXYZ>(cloud, color_handler, "cloud");

        // pcl::PointXYZ position(0, 0, 0); // Just for visualization
        // {
        //     std::lock_guard<std::mutex> lock(pfh_mutex);
        //     for (size_t i = 0; i < pfh_features_->size(); ++i)
        //     {
        //         pcl::PointXYZ arrow_start(position);
        //         pcl::PointXYZ arrow_end(position.x + 0.1, position.y + 0.01, position.z + 0.01);

        //         viewer->addArrow(arrow_start, arrow_end, 0.0, 1.0, 0.0, std::to_string(i));
        //     }
        // }

        counter++;

        // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        // pcl::copyPointCloud(*cloud, *colored_cloud);

        // for (size_t j = 0; j < colored_cloud->size(); ++j)
        // {
        //     pcl::PointXYZRGB &point = colored_cloud->at(j);
        //     pcl::FPFHSignature33 &feature = fpfh_features->at(j);
        //     uint8_t r = static_cast<uint8_t>(255 * feature.histogram[0]); // Adjust as needed
        //     uint8_t g = static_cast<uint8_t>(255 * feature.histogram[1]); // Adjust as needed
        //     uint8_t b = static_cast<uint8_t>(255 * feature.histogram[2]); // Adjust as needed
        //     point.r = r;
        //     point.g = g;
        //     point.b = b;
        // }
        // std::string cloud_name = "fpfh_colored_cloud" + std::to_string(counter);
        // viewer->addCoordinateSystem(5.0);
        // viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, cloud_name);
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
        while (ros::ok() && !viewer->wasStopped())
        {
            ros::spinOnce();
            viewer->spinOnce(100);
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Update every 10 ms
            ros::Duration(0.1).sleep();                                  // Sleep for a short duration to avoid high CPU usage
        }
    }
    void timerCB(const ros::TimerEvent &)
    {
        if (viewer->wasStopped())
        {
            ros::shutdown();
        }
    }

private:
    void computePFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const pcl::PointCloud<pcl::Normal>::Ptr &cloud_normals)
    {
        pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_features(new pcl::PointCloud<pcl::PFHSignature125>);

        pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh_estimation;
        pfh_estimation.setInputCloud(cloud);
        pfh_estimation.setInputNormals(cloud_normals);
        pfh_estimation.setRadiusSearch(20); // Adjust as needed
        pfh_estimation.compute(*pfh_features);

        std::lock_guard<std::mutex> lock(pfh_mutex);
        pfh_features_ = pfh_features;
    }
    pcl::PointCloud<pcl::PointXYZI> accumulated_cloud;
    // pcl::visualization::CloudViewer viewer;
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::ServiceServer save_service;
    ros::Timer viewer_timer;
    int counter;
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_features_;
    std::mutex pfh_mutex;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pcd_converter_node");
    PcdConverter subscriber;
    subscriber.spin();
    subscriber.save_to_pcd();
    return 0;
}