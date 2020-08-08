#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <cmath>
#include <vector>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/surface/impl/mls.hpp>
#include <pcl/surface/impl/gp3.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <algorithm>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include "ground_removal.hpp"

template<typename TPointType>
typename pcl::PointCloud<TPointType>::Ptr vector_to_pointcloud(std::vector<std::vector<float >> pc) {
    typename pcl::PointCloud<TPointType>::Ptr pcl_pc(new pcl::PointCloud<TPointType>);
    for (auto &i : pc) {
        TPointType point;
        point.x = i[0];
        point.y = i[1];
        point.z = i[2];
        point.intensity = i[3] * 255;

        pcl_pc->points.push_back(point);
    }
    return pcl_pc;
}

/**
 * visulize pointcloud
 * @param cloud
 */
void visualize(PointCloudPtr cloud) {
    // vis
    pcl::visualization::CloudViewer viewer("Demo viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped()) {}
}

/**
 * 分割出视野ROI
 * @param input
 * @param fov
 * @return
 */
PointCloudPtr crop_ROI(const PointCloudPtr &input, float fov, float forward_distance = 200, float height_limit = 8) {

    // filter
    pcl::PassThrough<XYZI> pass;
    pass.setInputCloud(input);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, forward_distance);
    pass.filter(*input);

    pass.setInputCloud(input);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-2, height_limit);
    pass.filter(*input);


    PointCloudPtr output(new PointCloud);
    for (auto &p:*input) {
        if (abs(atan(p.y / p.x)) < fov) {
            output->push_back(p);
        }
    }
    return output;
}


/**
 * load KITTI bin format pointcloud
 * @param path file path
 * @return
 */
std::vector<std::vector<float>> load_KITTI_pointcloud(const std::string &path) {
    std::ifstream fin;
    fin.open(path, std::ios::in | std::ios::binary);
    std::vector<std::vector<float>> pointcloud;
    std::vector<float> point;

    const int data_stride = 4;
    int data_cnt = 0;
    float data;
    while (fin.peek() != EOF) {
        // read 4 bytes of data
        fin.read(reinterpret_cast<char *> (&data), sizeof(float));
        point.push_back(data);

        if (++data_cnt % data_stride == 0) {
            data_cnt = 0;
            pointcloud.push_back(point);
            point.clear();
        }
    }
    fin.close();
    return pointcloud;
}


int main(int argc, char *argv[]) {

    // load pointcloud
    auto cloud = vector_to_pointcloud<XYZI>(load_KITTI_pointcloud(std::string(argv[1])));
//    visualize(cloud);

    FilterGroundResult segmentation = filter_ground(cloud, 0.2, 20);
    auto landscape = segmentation.non_ground, ground = segmentation.ground;

    visualize(landscape);

    visualize(ground);

    return 0;
}