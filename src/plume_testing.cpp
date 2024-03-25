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
        ROS_INFO("adding preproc sonar data");

        counter++;
    }
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::fromROSMsg(*msg, *cloud);
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(cloud);

        
        
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