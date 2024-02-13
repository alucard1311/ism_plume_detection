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
#include <thread>
#include <chrono>

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

        // ROS_INFO("Processing Data");

        pcl::fromROSMsg(*msg, *cloud);

        uniform_sampling.setInputCloud(cloud);
        uniform_sampling.setRadiusSearch(0.4);
        uniform_sampling.filter(*cloud);

        vg.setInputCloud(cloud);
        vg.setLeafSize(0.4, 0.4, 0.01f); // Adjust based on your requirements
        vg.filter(*cloud);

        // Surface Normal Estimation

        // Adaptive Radii
        double radii = 5;
        // radii =computeAdaptiveRadius(cloud,5);

        ROS_INFO("Global adaptive radius: %f", radii);

        normal_estimator.setInputCloud(cloud);
        normal_estimator.setNumberOfThreads(omp_get_max_threads());
        normal_estimator.setKSearch(30);
        normal_estimator.compute(*cloud_normals);
        ROS_INFO("Done Computing Normals Data");
        std::string cloud_name = "cloud" + std::to_string(counter);
        viewer->addPointCloud(cloud, cloud_name);
        std::string normals_name = "normals" + std::to_string(counter);

        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 10, 0.02, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 30, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0, 0.0, 0.0, normals_name);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //        ROS_INFO("Starting FPFH Estimation");

        fpfh_estimation.setInputCloud(cloud);
        fpfh_estimation.setInputNormals(cloud_normals);

        // Correctly using the KdTree with the FPFH estimation object
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        fpfh_estimation.setSearchMethod(tree);
        fpfh_estimation.setRadiusSearch(40); // Make sure this radius is appropriate for your data
        fpfh_estimation.setNumberOfThreads(omp_get_max_threads());
        ROS_INFO("I am starting to compute features");

        // Pre-filtering non-finite normals
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::Normal>::Ptr filtered_normals(new pcl::PointCloud<pcl::Normal>);
        for (size_t i = 0; i < cloud_normals->size(); ++i)
        {
            if (pcl::isFinite<pcl::Normal>((*cloud_normals)[i]))
            {
                filtered_cloud->push_back((*cloud)[i]);
                filtered_normals->push_back((*cloud_normals)[i]);
            }
            else
            {
                PCL_WARN("Normal at index %zu is not finite\n", i);
            }
        }

        // Use the filtered cloud and normals for FPFH computation
        fpfh_estimation.setInputCloud(filtered_cloud);
        fpfh_estimation.setInputNormals(filtered_normals);
        fpfh_estimation.compute(*fpfh_features);
        ROS_INFO("Done Computing Features");

        int index = 0; // For example, visualize the histogram for the first point

        // Add the histogram data to the visualizer
        std::string vis_name = "fpfh_histogram" + std::to_string(counter);

        visualizer.addFeatureHistogram(*fpfh_features, 33, vis_name, 300, 400);

        // pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot_estimation; // Create a SHOTEstimation object
        // shot_estimation.setInputCloud(cloud);
        // shot_estimation.setInputNormals(cloud_normals);
        // shot_estimation.setRadiusSearch(30); // Set the radius for SHOT estimation
        // // Note: The search radius for SHOT can be different from FPFH

        // shot_estimation.compute(*shot_features);

        // int index = 0; // For example, visualize the histogram for the first point

        // // Add the histogram data to the visualizer
        // std::string vis_name = "fpfh_histogram" + std::to_string(counter);

        // // Assuming you have a pcl::PointCloud<pcl::SHOT352>::Ptr named shot_features
        // if (!shot_features->empty())
        // {
        //     // Access the descriptor values for the first point as an example
        //     int index = 0; // Index of the point you're interested in

        //     // Make sure the index is within the range of available points
        //     if (index < shot_features->points.size())
        //     {
        //         // Create a vector to hold the descriptor values for the histogram
        //         std::vector<float> descriptor_values;

        //         for (int d = 0; d < pcl::SHOT352::descriptorSize(); ++d)
        //         {
        //             float value = shot_features->points[index].descriptor[d];
        //             descriptor_values.push_back(value); // Add the value to the vector
        //         }

        //         // Create and configure the histogram visualizer
        //         pcl::visualization::PCLHistogramVisualizer visualizer;
        //         visualizer.setBackgroundColor(255, 255, 255); // Set a white background for better visibility

        //         // Add the histogram data to the visualizer
        //         visualizer.addFeatureHistogram<pcl::SHOT352>(descriptor_values, pcl::SHOT352::descriptorSize(), "SHOT Histogram", 300, 300);

        //         // Display the histogram
        //         visualizer.spin();
        //     }
        //     else
        //     {
        //         ROS_WARN("Specified point index is out of bounds.");
        //     }
        // }
        // else
        // {
        //     ROS_WARN("SHOT features are empty. Cannot access descriptor values.");
        // }

        // Spin to keep the visualization window open

        counter++;
    }
    double computeAdaptiveRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int k)
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud);

        double total_average_distance = 0.0;
        int valid_points = 0;

        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            std::vector<int> indices(k);
            std::vector<float> sqr_distances(k);
            double average_distance = 0.0;

            if (kdtree.nearestKSearch(cloud->points[i], k, indices, sqr_distances) > 0)
            {
                for (float sqr_distance : sqr_distances)
                {
                    average_distance += std::sqrt(sqr_distance);
                }
                average_distance /= static_cast<double>(k);
                total_average_distance += average_distance;
                valid_points++;
            }
        }

        if (valid_points > 0)
        {
            double global_average = total_average_distance / valid_points;
            return 2 * global_average; // Double the global average distance to get the adaptive radius
        }
        else
        {
            return 0.0; // Return 0 or an appropriate default value if no valid points were found
        }
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