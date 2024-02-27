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

        // Surface Normal Estimation

        normal_estimator.setInputCloud(cloud);
        normal_estimator.setNumberOfThreads(omp_get_max_threads());
        normal_estimator.setKSearch(5);
        normal_estimator.compute(*cloud_normals);
        ROS_INFO("Done Computing Normals Data");
        std::string cloud_name = "cloud" + std::to_string(counter);
        //viewer->addPointCloud(cloud, cloud_name);
        std::string normals_name = "normals" + std::to_string(counter);

        // viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, cloud_normals, 10, 0.02, normals_name);
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 30, normals_name);
        // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0, 0.0, 0.0, normals_name);
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //        ROS_INFO("Starting FPFH Estimation");

        // Correctly using the KdTree with the FPFH estimation object
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        fpfh_estimation.setSearchMethod(tree);
        fpfh_estimation.setRadiusSearch(10); // Make sure this radius is appropriate for your data
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
        // Eigen::VectorXf meanHistogram = computeMeanFPFHHistogram(fpfh_features);
        // ROS_INFO("Done Computing Mean Histograms");
        // // visualizeHistogram(meanHistogram);
        // addHistogramToQueue(meanHistogram);
        Eigen::MatrixXf fpfhMatrix(fpfh_features->points.size(), 33);
        for (size_t i = 0; i < fpfh_features->points.size(); ++i)
        {
            for (int j = 0; j < 33; ++j)
            {
                fpfhMatrix(i, j) = fpfh_features->points[i].histogram[j];
            }
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr fpfh_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // Fill the point cloud using the matrix data
        for (int i = 0; i < fpfhMatrix.cols(); ++i)
        {
            pcl::PointXYZ point;
            point.x = fpfhMatrix(0, i); // X coordinate
            point.y = fpfhMatrix(1, i); // Y coordinate
            point.z = fpfhMatrix(2, i); // Z coordinate
            cloud->push_back(point);
        }
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(fpfh_cloud);
        Eigen::VectorXf mean = fpfhMatrix.colwise().mean();
        Eigen::MatrixXf centeredMatrix = fpfhMatrix.rowwise() - mean.transpose();

        // Step 2: Compute the covariance matrix
        Eigen::MatrixXf covarianceMatrix = centeredMatrix.transpose() * centeredMatrix / float(centeredMatrix.rows() - 1);

        // Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigenSolver(covarianceMatrix);
        Eigen::MatrixXf eigenVectors = eigenSolver.eigenvectors();

        // Step 4: Project the centered data onto the PCA space
        Eigen::MatrixXf transformedFeatures = centeredMatrix * eigenVectors;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcaCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (size_t i = 0; i < transformedFeatures.rows(); ++i)
        {
            pcl::PointXYZRGB point;
            point.x = transformedFeatures(i, 0); // First principal component
            point.y = transformedFeatures(i, 1); // Second principal component
            point.z = transformedFeatures(i, 2); // Third principal component
            // Assign a color based on the first component for visualization
            uint8_t intensity = static_cast<uint8_t>(point.x * 255);
            uint32_t rgb = (static_cast<uint32_t>(intensity) << 16 |
                            static_cast<uint32_t>(intensity) << 8 | static_cast<uint32_t>(intensity));
            point.rgb = *reinterpret_cast<float *>(&rgb);
            pcaCloud->points.push_back(point);
        }
        std::string pca_cloud_name = "pca_cloud" + std::to_string(counter);
        //viewer->addPointCloud<pcl::PointXYZRGB>(pcaCloud,pca_cloud_name);

        counter++;
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