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
#include <pcl/common/common.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <visualization_msgs/Marker.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <limits>
#include <pcl/segmentation/sac_segmentation.h>

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
        nh.setParam("tcpNoDelay", true);
        sub = nh.subscribe("/bsd_sonar/pcl_postproc_sonar", 1, &PcdConverter::pc_callback, this);
        data_sub = nh.subscribe("/bsd_sonar/pcl_preproc_sonar", 1, &PcdConverter::sonar_callback, this);
        bb_pub = nh.advertise<visualization_msgs::MarkerArray>("bb_ransac", 1000);
        // vertical_axis_pub = nh.advertise<visualization_msgs::Marker>("vertical_axis_marker", 1);

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

        // pcl::fromROSMsg(*msg, *cloud);

        // uniform_sampling.setInputCloud(cloud);
        // uniform_sampling.setRadiusSearch(0.4);
        // uniform_sampling.filter(*cloud);

        // vg.setInputCloud(cloud);
        // vg.setLeafSize(0.4, 0.4, 0.01f); // Adjust based on your requirements
        // vg.filter(*cloud);

        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 255, 255);
        // std::string cloud_name = "pre_cloud_" + std::to_string(counter);

        // viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, cloud_name);
        // ROS_INFO("adding preproc sonar data");

        counter++;
    }
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::fromROSMsg(*msg, *cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr remainingCloud(new pcl::PointCloud<pcl::PointXYZ>(*cloud));
        visualization_msgs::MarkerArray marker_array;
        int lineCount = 0;

        while (remainingCloud->points.size() > 10)
        {
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_LINE);
            seg.setMethodType(pcl::SAC_RANSAC);
            //seg.setMaxIterations(100);
            seg.setDistanceThreshold(0.005); // Set this according to your application
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

            seg.setInputCloud(remainingCloud);
            seg.segment(*inliers, *coefficients);

            if (inliers->indices.empty())
            {
                break;
            }

            // Extract inliers
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(remainingCloud);
            extract.setIndices(inliers);
            extract.setNegative(false);

            pcl::PointCloud<pcl::PointXYZ>::Ptr inlierCloud(new pcl::PointCloud<pcl::PointXYZ>());
            extract.filter(*inlierCloud);

            // Compute bounding box for inliers (simplified to min/max Z)
            float minZ = std::numeric_limits<float>::max(), maxZ = -std::numeric_limits<float>::max();
            for (const auto &point : inlierCloud->points)
            {
                minZ = std::min(minZ, point.z);
                maxZ = std::max(maxZ, point.z);
            }

            // Create a marker for the bounding box
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map"; // Set to your point cloud's frame ID
            marker.header.stamp = ros::Time::now();
            marker.ns = "ransac_bb";
            marker.id = counter+(lineCount++);
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = coefficients->values[0];
            marker.pose.position.y = coefficients->values[1];
            marker.pose.position.z = (minZ + maxZ) / 2.0;
            marker.scale.x = 0.1;         // Set the width of the line (assuming a small constant value)
            marker.scale.y = 0.1;         // Set the thickness of the line (assuming a small constant value)
            marker.scale.z = maxZ - minZ; // Height of the bounding box
            marker.color.a = 0.3;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 1.0;

            marker_array.markers.push_back(marker);
            extract.setNegative(true);
            pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
            extract.filter(*tempCloud);
            remainingCloud.swap(tempCloud);
        }

        // Publish the marker array containing all bounding boxes
        ROS_INFO("Publishing the bouding boxes");
        bb_pub.publish(marker_array);

        counter++;
    }

    // void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    // {
    //     pcl::fromROSMsg(*msg, *cloud);

    //     pcl::PCA<pcl::PointXYZ> pca;
    //     pca.setInputCloud(cloud);

    //     Eigen::Vector3f eigenValues = pca.getEigenValues();
    //     Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

    //     Eigen::Vector3f principalDirection = eigenVectors.col(0);

    //     // Visualization of the point cloud with principal direction vector
    //     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    //     std::string cloud_name = "cloud_" + std::to_string(counter);

    //     viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, cloud_name);
    //     ROS_INFO("adding Directional Vector");
    //     pcl::PointXYZ center(pca.getMean().x(), pca.getMean().y(), pca.getMean().z());

    //     // Increase the size of the arrow for the principal direction vector
    //     double arrowScale = 3; // Adjust this value to change the size of the arrow
    //     pcl::PointXYZ principalPoint(center.x + principalDirection.x() * arrowScale, center.y + principalDirection.y() * arrowScale, center.z + principalDirection.z() * arrowScale);
    //     std::string arrow_name = "arrow_" + std::to_string(counter);
    //     viewer->addArrow(principalPoint, center, 1.0, 0.0, 0.0, false, arrow_name);

    //     // Adding a coordinate system at the base of the principal direction arrow
    //     double coordinateSystemScale = 15; // Adjust this value to change the size of the coordinate system

    //     viewer->addCoordinateSystem(coordinateSystemScale, center.x, center.y, center.z, arrow_name + "axis_name");

    //     counter++;
    // }

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
    ros::Publisher bb_pub;

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