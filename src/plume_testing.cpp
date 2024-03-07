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
        save_service = nh.advertiseService("avl/save_point_cloud", &PcdConverter::saveServiceCallback, this);
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
    void pc_callback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {

        // ROS_INFO("Processing Data");

        pcl::fromROSMsg(*msg, *cloud);

        uniform_sampling.setInputCloud(cloud);
        uniform_sampling.setRadiusSearch(0.4);
        uniform_sampling.filter(*cloud);

        vg.setInputCloud(cloud);
        vg.setLeafSize(0.4, 0.4, 0.01f); // Adjust based on your requirements
        vg.filter(*cloud);

        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_parts(4);
        for (int i = 0; i < 4; ++i)
        {
            cloud_parts[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        }
        splitCloudIntoParts(*cloud, cloud_parts);
        std::string cloud_name = "cloud_" + std::to_string(counter);

        // Visualize each part
        for (int i = 0; i < cloud_parts.size(); ++i)
        {
            viewer->addPointCloud<pcl::PointXYZ>(cloud_parts[i], pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ>(cloud_parts[i]), cloud_name + std::to_string(i));
        }

        viewer->addCoordinateSystem(1.0);
        viewer->initCameraParameters();
        counter++;
    }


    void splitCloudIntoParts(const pcl::PointCloud<pcl::PointXYZ>& input_cloud, std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& cloud_parts)
{
    // Sort points based on Z-axis
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sorted(new pcl::PointCloud<pcl::PointXYZ>);
    *cloud_sorted = input_cloud;
    std::sort(cloud_sorted->points.begin(), cloud_sorted->points.end(), [](const pcl::PointXYZ& a, const pcl::PointXYZ& b) {
        return a.z < b.z;
    });

    // Calculate the number of points in each part
    size_t part_size = cloud_sorted->size() / cloud_parts.size();

    // Extract points for each part
    for (size_t i = 0; i < cloud_parts.size(); ++i) {
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        for (size_t j = i * part_size; j < (i + 1) * part_size && j < cloud_sorted->size(); ++j) {
            indices->indices.push_back(j);
        }

        // Last part gets any remaining points due to integer division
        if (i == cloud_parts.size() - 1) {
            for (size_t j = (i + 1) * part_size; j < cloud_sorted->size(); ++j) {
                indices->indices.push_back(j);
            }
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_sorted);
        extract.setIndices(indices);
        extract.setNegative(false);
        extract.filter(*cloud_parts[i]);
    }
}
    Eigen::VectorXf computeMeanFPFHHistogram(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr &fpfh_features)
    {
        if (fpfh_features->empty())
        {
            std::cerr << "FPFH features are empty." << std::endl;
            return Eigen::VectorXf();
        }

        // Initialize a vector to hold the sum of all histograms
        Eigen::VectorXf histogramSum = Eigen::VectorXf::Zero(pcl::FPFHSignature33::descriptorSize());

        // Accumulate all FPFH histograms
        for (const auto &feature : fpfh_features->points)
        {
            for (int i = 0; i < pcl::FPFHSignature33::descriptorSize(); ++i)
            {
                histogramSum[i] += feature.histogram[i];
            }
        }

        // Compute the mean histogram
        Eigen::VectorXf meanHistogram = histogramSum / static_cast<float>(fpfh_features->size());

        return meanHistogram;
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
    ros::ServiceServer save_service;
    ros::Timer viewer_timer;
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
    ros::init(argc, argv, "pcd_converter_node");
    PcdConverter subscriber;
    subscriber.spin();
    subscriber.save_to_pcd();
    return 0;
}