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


ros::Publisher pcl_pub;



int ism_model(){
    std::vector<std::string> training_files = {
        "/home/cmar/CMAR/pcd_traning_files/traning_batch_1649760492.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_424238335.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_1681692777.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_596516649.pcd",
        // "/home/cmar/CMAR/pcd_traning_files/traning_batch_846930886.pcd",
        // Add more training file paths as needed
    };

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> training_clouds;
    std::vector<pcl::PointCloud<pcl::Normal>::Ptr> training_normals;
    std::vector<unsigned int> training_classes;

    for (const std::string &training_file : training_files)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(training_file, *cloud) == -1)
        {
            PCL_ERROR("Couldn't read file %s\n", training_file.c_str());
            continue; // Skip this file and continue with the next one
        }

        // Compute Normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud);
        normal_estimator.setRadiusSearch(0.03);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.compute(*cloud_normals);

        unsigned int tr_class = 1; // Set the class label for this training file

        training_clouds.push_back(cloud);
        training_normals.push_back(cloud_normals);
        training_classes.push_back(tr_class);
    }

    if (training_clouds.empty())
    {
        std::cerr << "No valid training data loaded. Exiting." << std::endl;
        return -1;
    }

    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>>::Ptr fpfh(new pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153>>);
    fpfh->setRadiusSearch(30.0);
    pcl::Feature<pcl::PointXYZ, pcl::Histogram<153>>::Ptr feature_estimator(fpfh);

    pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal> ism;
    ism.setFeatureEstimator(feature_estimator);
    ism.setTrainingClouds(training_clouds);
    ism.setTrainingNormals(training_normals);
    ism.setTrainingClasses(training_classes);
    ism.setSamplingSize(2.0f);

    pcl::ism::ImplicitShapeModelEstimation<153, pcl::PointXYZ, pcl::Normal>::ISMModelPtr model(new pcl::features::ISMModel);
    ism.trainISM(model);

    std::string file("trained_ism_model_2.txt");
    model->saveModelToFile(file);

    /////////////////////////////////////////////////////////////////////////////////////////////////
    std::string testing_file("/home/cmar/CMAR/pcd_traning_files/traning_batch_846930886.pcd");

    unsigned int testing_class = 1;
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr testing_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(testing_file, *testing_cloud) == -1)
    {
        PCL_ERROR("Couldn't read file %s\n", testing_file.c_str());
        return -1; // Skip this file and continue with the next one
    }

    
    pcl::PointCloud<pcl::Normal>::Ptr testing_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
    normal_estimator.setInputCloud(testing_cloud);
    normal_estimator.setRadiusSearch(10);
    normal_estimator.compute(*testing_normals);   
    

    pcl::features::ISMVoteList<pcl::PointXYZ>::Ptr vote_list = ism.findObjects(
        model,
        testing_cloud,
        testing_normals,
        testing_class);


    ROS_INFO("I am here");

    double radius = model->sigmas_[testing_class];
    double sigma = model->sigmas_[testing_class];
    std::vector<pcl::ISMPeak, Eigen::aligned_allocator<pcl::ISMPeak>> strongest_peaks;
    vote_list->findStrongestPeaks(strongest_peaks, testing_class, radius, sigma);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    colored_cloud->height = 0;
    colored_cloud->width = 1;

    pcl::PointXYZRGB point;
    point.r = 255;
    point.g = 255;
    point.b = 255;

    for (std::size_t i_point = 0; i_point < testing_cloud->size(); i_point++)
    {
        point.x = (*testing_cloud)[i_point].x;
        point.y = (*testing_cloud)[i_point].y;
        point.z = (*testing_cloud)[i_point].z;
        colored_cloud->points.push_back(point);
    }
    colored_cloud->height += testing_cloud->size();

    point.r = 255;
    point.g = 0;
    point.b = 0;
    for (std::size_t i_vote = 0; i_vote < strongest_peaks.size(); i_vote++)
    {
        point.x = strongest_peaks[i_vote].x;
        point.y = strongest_peaks[i_vote].y;
        point.z = strongest_peaks[i_vote].z;
        colored_cloud->points.push_back(point);
    }
    colored_cloud->height += strongest_peaks.size();

    sensor_msgs::PointCloud2 colored_cloud_msg;
    pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
    colored_cloud_msg.header.frame_id = "/base_link";
    colored_cloud_msg.
    pcl_pub.publish(colored_cloud_msg);


}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ism_node");
    ros::NodeHandle nh;
    pcl_pub = nh.advertise<sensor_msgs::PointCloud2>("/avl/ism_pointclouds", 1000);
    ism_model();

    // pcl_postproc_pub.publish(colored_cloud);
    ros::spin();
    
    return 0;
}