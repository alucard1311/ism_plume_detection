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
#include <pcl/visualization/pcl_visualizer.h>

#include <thread>
#include <chrono>


#include "plume_detection/Histogram.h"  // Replace `your_package_name` with the name of your package


ros::Publisher pcl_pub;
pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Point Cloud with Normals"));

int ism_model()
{
    std::vector<std::string> training_files = {
        "/home/cmar/avl/src/plume_detection/src/point_cloud_4.pcd",
        //"/home/cmar/CMAR/pcd_traning_files/traning_batch_424238335.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_1681692777.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_596516649.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_846930886.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_719885386.pcd",
        //  "/home/cmar/CMAR/pcd_traning_files/traning_batch_1714636915.pcd",
        //  "/home/cmar/CMAR/pcd_traning_files/traning_batch_1804289383.pcd",
        //   "/home/cmar/CMAR/pcd_traning_files/traning_batch_1957747793.pcd",

        // Add more training file paths as needed
    };

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
    std::vector<unsigned int> training_classes;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ROS_INFO("I am STARTING computing Normals");
    for (const std::string &training_file : training_files)
    {

        if (pcl::io::loadPCDFile<pcl::PointXYZ>(training_file, *cloud) == -1)
        {
            PCL_ERROR("Couldn't read file %s\n", training_file.c_str());
            continue; // Skip this file and continue with the next one
        }

        // Compute Normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud);
        normal_estimator.setRadiusSearch(1);

        normal_estimator.compute(*cloud_normals);
       

        unsigned int tr_class = 1; // Set the class label for this training file

        training_clouds.push_back(cloud);
        training_normals.push_back(cloud_normals);
        training_classes.push_back(tr_class);
    }

    for (size_t i = 0; i < training_clouds.size(); ++i)
    {
        std::string cloud_name = "cloud" + std::to_string(i);
        viewer->addPointCloud(training_clouds[i], cloud_name);
    }

    for (size_t i = 0; i < training_normals.size(); ++i)
    {
        std::string normals_name = "normals" + std::to_string(i);
        // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(training_clouds[i], 0, 255, 0); // Green color
        viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(training_clouds[i], training_normals[i], 10, 0.02, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 30, normals_name);
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0, 0.0, 0.0, normals_name);
    }
    ROS_INFO("I am done visualising Normals, starting fpfh estimation");
    /////////////////////////////////////////////////////////

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setInputCloud(cloud);           // Assuming you're working with a single cloud
    fpfh_estimation.setInputNormals(cloud_normals); // Assuming you've computed normals
    fpfh_estimation.setRadiusSearch(2);             // Adjust the radius based on your data
    fpfh_estimation.compute(*fpfh_features);
    ROS_INFO("I am done Computing FPFH features, starting visualising fpfh estimation");
    /////////////////////////////////////////////////////////
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, *colored_cloud);

    for (size_t j = 0; j < colored_cloud->size(); ++j)
    {
        pcl::PointXYZRGB &point = colored_cloud->at(j);
        pcl::FPFHSignature33 &feature = fpfh_features->at(j);
        uint8_t r = static_cast<uint8_t>(255 * feature.histogram[0]); // Adjust as needed
        uint8_t g = static_cast<uint8_t>(255 * feature.histogram[1]); // Adjust as needed
        uint8_t b = static_cast<uint8_t>(255 * feature.histogram[2]); // Adjust as needed
        point.r = r;
        point.g = g;
        point.b = b;
    }
    std::string cloud_name = "fpfh_colored_cloud";
    viewer->addCoordinateSystem(5.0);
    viewer->addPointCloud<pcl::PointXYZRGB>(colored_cloud, cloud_name);

    // std::string fpfh_name = "fpfh_features";
    // viewer->addPointCloud<pcl::FPFHSignature33>(fpfh_features, fpfh_name);

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>>::Ptr fpfh(new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>>);
    // fpfh->setRadiusSearch(10.0);
    // pcl::Feature<pcl::PointXYZ, pcl::Histogram<153>>::Ptr feature_estimator(fpfh);

    // pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism;
    // ism.setFeatureEstimator(feature_estimator);
    // ism.setTrainingClouds(training_clouds);
    // ism.setTrainingNormals(training_normals);
    // ism.setTrainingClasses(training_classes);
    // ism.setSamplingSize(2.0f);

    // pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model(new pcl::features::ISMModel);

    // ism.trainISM(model);

    // std::string file("trained_ism_model_all.txt");
    // ROS_INFO("Saving the trained mdoel");
    // model->saveModelToFile(file);

    // std::string pcd_file ("trained_ism_model_all.txt");
    // model->loadModelFromfile(pcd_file);
    // ROS_INFO("Model Loaded");

    // /////////////////////////////////////////////////////////////////////////////////////////////////
    // //std::string testing_file("/home/cmar/CMAR/pcd_traning_files/traning_batch_596516649.pcd");
    // std::string testing_file("/home/cmar/avl/src/plume_detection/src/point_cloud_3.pcd");

    // unsigned int testing_class = 1;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // if (pcl::io::loadPCDFile<pcl::PointXYZ>(testing_file, *testing_cloud) == -1)
    // {
    //     //ROS_INFO("Openning the file");
    //     PCL_ERROR("Couldn't read file %s\n", testing_file.c_str());
    //     return -1; // Skip this file and continue with the next one
    // }
    // ROS_INFO("Testing PCD Loaded");

    // pcl::PointCloud<pcl::Normal>::Ptr testing_normals(new pcl::PointCloud<pcl::Normal>);
    // pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    // normal_estimator.setInputCloud(testing_cloud);
    // normal_estimator.setRadiusSearch(10);
    // normal_estimator.compute(*testing_normals);

    // ROS_INFO("Classifying the Input Point Cloud");
    // pcl::features::ISMVoteList<pcl::PointXYZ>::Ptr vote_list = ism.findObjects(
    //     model,
    //     testing_cloud,
    //     testing_normals,
    //     testing_class);
    // ROS_INFO("Completed Classification");
    // double radius = model->sigmas_[testing_class];
    // double sigma = model->sigmas_[testing_class];
    // std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak>> strongest_peaks;
    // vote_list->findStrongestPeaks(strongest_peaks, testing_class, radius, sigma);

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // colored_cloud->height = 0;
    // colored_cloud->width = 1;

    // pcl::PointXYZRGB point;
    // point.r = 255;
    // point.g = 255;
    // point.b = 255;

    // for (std::size_t i_point = 0; i_point < testing_cloud->size(); i_point++)
    // {
    //     point.x = (*testing_cloud)[i_point].x;
    //     point.y = (*testing_cloud)[i_point].y;
    //     point.z = (*testing_cloud)[i_point].z;
    //     colored_cloud->points.push_back(point);
    // }
    // colored_cloud->height += testing_cloud->size();

    // point.r = 255;
    // point.g = 0;
    // point.b = 0;
    // for (std::size_t i_vote = 0; i_vote < strongest_peaks.size(); i_vote++)
    // {
    //     point.x = strongest_peaks[i_vote].x;
    //     point.y = strongest_peaks[i_vote].y;
    //     point.z = strongest_peaks[i_vote].z;
    //     colored_cloud->points.push_back(point);
    // }
    // colored_cloud->height += strongest_peaks.size();

    // sensor_msgs::PointCloud2 colored_cloud_msg;
    // pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
    // pcl::io::savePCDFileASCII("classification_output.pcd", *colored_cloud);
    // ROS_INFO("Saved Point PointCloud");
    // // ros::Time::init();
    // colored_cloud_msg.header.stamp = ros::Time::now();
    // colored_cloud_msg.header.frame_id = "/base_link";
    // pcl_pub.publish(colored_cloud_msg);
}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "ism_node");
    ros::NodeHandle nh;
    ros::Rate rate(10);
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/avl/ism_pointclouds", 1000);
    ism_model();

    // pcl_postproc_pub.publish(colored_cloud);
    // ros::spinOnce();
    // rate.sleep();

    return 0;
}