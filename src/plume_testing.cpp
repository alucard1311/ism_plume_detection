#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/feature.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/impl/fpfh.hpp>
#include <pcl/recognition/implicit_shape_model.h>
#include <pcl/recognition/impl/implicit_shape_model.hpp>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <std_srvs/Trigger.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/pfh.h>
#include <pcl/point_types.h>
#include <mutex>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <Eigen/Dense>
#include <vector>
#include <condition_variable>
#include <pcl/common/pca.h>
#include <visualization_msgs/Marker.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>

pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud with Normals"));
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
pcl::PointCloud<pcl::PointXYZI> accumulated_cloud;
pcl::UniformSampling<pcl::PointXYZ> uniform_sampling;
pcl::VoxelGrid<pcl::PointXYZ> vg;
pcl::visualization::PCLHistogramVisualizer visualizer;

pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
pcl::PointCloud<pcl::SHOT352>::Ptr shot_features{new pcl::PointCloud<pcl::SHOT352>};
std::queue<Eigen::VectorXf> histogramQueue;
std::mutex queueMutex;
std::condition_variable queueCondVar;

class PcdConverter
{
public:
    PcdConverter() : counter(0), finished(false)
    {
        sub = nh.subscribe("/bsd_sonar/pcl_postproc_sonar", 1, &PcdConverter::pc_callback, this);
        data_sub = nh.subscribe("/bsd_sonar/pcl_preproc_sonar", 1, &PcdConverter::sonar_callback, this);
        principal_direction_pub = nh.advertise<visualization_msgs::Marker>("principal_direction_marker", 1);
        vertical_axis_pub = nh.advertise<visualization_msgs::Marker>("vertical_axis_marker", 1);

        viewer_timer = nh.createTimer(ros::Duration(1), &PcdConverter::timerCB, this);
    }
    ~PcdConverter()
    {
        // Signal the visualization thread to finish and wake it up
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            finished = true;
        }
        queueCondVar.notify_one();

        // Wait for the visualization thread to finish
        if (visThread.joinable())
        {
            visThread.join();
        }
    }
    void sonar_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {

        // ROS_INFO("Processing Data");

        pcl::fromROSMsg(*msg, *cloud);

        // uniform_sampling.setInputCloud(cloud);
        // uniform_sampling.setRadiusSearch(0.4);
        // uniform_sampling.filter(*cloud);

        // vg.setInputCloud(cloud);
        // vg.setLeafSize(0.4, 0.4, 0.01f); // Adjust based on your requirements
        // vg.filter(*cloud);

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 255, 255);
        std::string cloud_name = "pre_cloud_" + std::to_string(counter);

        viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, cloud_name);
        ROS_INFO("adding preproc sonar data");

        counter++;
    }
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {

        // ROS_INFO("Processing Data");

        pcl::fromROSMsg(*msg, *cloud);

        // uniform_sampling.setInputCloud(cloud);
        // uniform_sampling.setRadiusSearch(0.4);
        // uniform_sampling.filter(*cloud);

        // vg.setInputCloud(cloud);
        // vg.setLeafSize(0.4, 0.4, 0.01f); // Adjust based on your requirements
        // vg.filter(*cloud);

        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(cloud);

        ROS_INFO("Getting EigenVectors and Values");
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

        Eigen::Vector3f principalDirection = eigenVectors.col(0);

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, centroid);

        visualization_msgs::Marker principal_direction_marker;
        principal_direction_marker.header.frame_id = "map"; // Set to your frame ID
        principal_direction_marker.header.stamp = ros::Time::now();
        principal_direction_marker.ns = "principal_direction";
        principal_direction_marker.id = 0;
        principal_direction_marker.type = visualization_msgs::Marker::ARROW;
        principal_direction_marker.action = visualization_msgs::Marker::ADD;
        principal_direction_marker.pose.position.x = centroid[0];
        principal_direction_marker.pose.position.y = centroid[1];
        principal_direction_marker.pose.position.z = centroid[2];
        principal_direction_marker.scale.x = 0.5; // Shaft diameter
        principal_direction_marker.scale.y = 0.1; // Head diameter
        principal_direction_marker.scale.z = 0;   // Head length, not applicable for arrows
        principal_direction_marker.color.a = 1.0; // Don't forget to set the alpha!
        principal_direction_marker.color.r = 1.0f;
        principal_direction_marker.color.g = 1.0f;
        principal_direction_marker.color.b = 1.0f;
        // Set the orientation of the marker to match the principal direction
        Eigen::Vector3f start(centroid[0], centroid[1], centroid[2]); // Using the centroid as start
        float arrowLength = 1.0;                                      // Set the desired arrow length
        Eigen::Vector3f end = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]) + principalDirection * arrowLength;

        Eigen::Quaternionf quat = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), (end - Eigen::Vector3f(centroid[0], centroid[1], centroid[2])));
        principal_direction_marker.pose.orientation.x = quat.x();
        principal_direction_marker.pose.orientation.y = quat.y();
        principal_direction_marker.pose.orientation.z = quat.z();
        principal_direction_marker.pose.orientation.w = quat.w();

        // Publish the marker
        principal_direction_pub.publish(principal_direction_marker);

        visualization_msgs::Marker vertical_marker;
        vertical_marker.header.frame_id = "map"; // Set to your frame ID, same as the principal direction marker
        vertical_marker.header.stamp = ros::Time::now();
        vertical_marker.ns = "vertical_axis";
        vertical_marker.id = counter; // Unique ID for this marker in the namespace
        vertical_marker.type = visualization_msgs::Marker::ARROW;
        vertical_marker.action = visualization_msgs::Marker::ADD;
        vertical_marker.pose.position.x = centroid[0];
        vertical_marker.pose.position.y = centroid[1];
        vertical_marker.pose.position.z = centroid[2];
        vertical_marker.scale.x = 0.5; // Shaft diameter, smaller than the principal direction for distinction
        vertical_marker.scale.y = 0.1; // Head diameter, smaller than the principal direction for distinction
        vertical_marker.scale.z = 0;   // Head length, not applicable for arrows
        vertical_marker.color.a = 1.0; // Don't forget to set the alpha!
        vertical_marker.color.r = 0.0;
        vertical_marker.color.g = 0.0;
        vertical_marker.color.b = 1.0; // Blue color for the vertical axis

        // The vertical axis is aligned with Z in most ROS coordinate systems
        Eigen::Vector3f verticalEnd = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]) + Eigen::Vector3f::UnitZ() * arrowLength; // Assuming Z is up
        Eigen::Quaternionf vertical_quat = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), (verticalEnd - Eigen::Vector3f(centroid[0], centroid[1], centroid[2])));
        vertical_marker.pose.orientation.x = vertical_quat.x();
        vertical_marker.pose.orientation.y = vertical_quat.y();
        vertical_marker.pose.orientation.z = vertical_quat.z();
        vertical_marker.pose.orientation.w = vertical_quat.w();

        // Publish the vertical axis marker
        vertical_axis_pub.publish(vertical_marker);

        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
        // std::string cloud_name = "cloud_" + std::to_string(counter);

        // viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, cloud_name);
        // ROS_INFO("adding Directional Vector");
        // pcl::PointXYZ center(pca.getMean().x(), pca.getMean().y(), pca.getMean().z());
        // std::string arrow_name = "arrow_" + std::to_string(counter);
        // pcl::PointXYZ principalPoint(center.x + principalDirection.x(), center.y + principalDirection.y(), center.z + principalDirection.z());
        // viewer->addArrow(principalPoint, center, 1.0, 0.0, 0.0, false, arrow_name);

        counter++;
    }

    void spin()
    {
        while (ros::ok() && !viewer->wasStopped())
        {
            ros::spinOnce();
            viewer->spinOnce(100);

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ros::Duration(0.1).sleep();
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
    // pcl::visualization::CloudViewer viewer;
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Subscriber data_sub;
    ros::ServiceServer save_service;
    ros::Timer viewer_timer;
    ros::Publisher principal_direction_pub;
    ros::Publisher vertical_axis_pub;

    int counter;
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_features_;
    std::thread visThread;                      // Visualization thread
    std::queue<Eigen::VectorXf> histogramQueue; // Queue for histograms to visualize
    std::mutex queueMutex;                      // Mutex for thread-safe access to the queue
    std::condition_variable queueCondVar;       // Condition variable for notifying the visualization thread
    bool finished;
};

int main(int argc, char **argv)
{
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
    ros::init(argc, argv, "pcd_converter_node");
    PcdConverter subscriber;
    subscriber.spin();
    return 0;
}