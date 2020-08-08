//
// Created by hviktortsoi on 20-5-27.
//
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <algorithm>
#include "ground_removal.hpp"

namespace py=pybind11;


py::array_t<float> ground_removal_kernel(
        const py::array_t<float> &input,
        double distance_th,
        int iter_times
) {
    auto ref_input = input.unchecked<2>();
    // 初始化pointcloud 数量是输入的numpy array中的point数量
    PointCloudPtr cloud(new PointCloud(ref_input.shape(0), 1));
//#pragma omp parallel for num_threads(NUM_THREADS)
    for (int i = 0; i < ref_input.shape(0); ++i) {
        cloud->points[i].x = ref_input(i, 0);
        cloud->points[i].y = ref_input(i, 1);
        cloud->points[i].z = ref_input(i, 2);
        cloud->points[i].intensity = ref_input(i, 3);
    }
//    std::cout << "INPUT SHAPE: " << ref_input.shape(0) << " " << ref_input.shape(1) << std::endl;
    std::cout << distance_th << " " << iter_times << std::endl;

    // filter ground
    FilterGroundResult segmentation = filter_ground(cloud, distance_th, iter_times);

    // results
    int data_field = 5;
    auto result = py::array_t<float>(py::array::ShapeContainer(
            {(const long) input.shape(0), data_field}
    ));
//    std::cout << "RESULT SHAPE: " << result.shape(0) << " " << result.shape(1) << std::endl;

    float *buf = (float *) result.request().ptr;
    // 非地面点
    for (int i = 0; i < segmentation.non_ground->size(); ++i) {
        int buf_index_base = i * data_field;
        buf[buf_index_base + 0] = segmentation.non_ground->points[i].x;
        buf[buf_index_base + 1] = segmentation.non_ground->points[i].y;
        buf[buf_index_base + 2] = segmentation.non_ground->points[i].z;
        buf[buf_index_base + 3] = segmentation.non_ground->points[i].intensity;
        buf[buf_index_base + 4] = 0.0;
    }
    // 地面点
    for (int i = 0; i < segmentation.ground->size(); ++i) {
        int buf_index_base = (segmentation.non_ground->size() + i) * data_field;
        buf[buf_index_base + 0] = segmentation.ground->points[i].x;
        buf[buf_index_base + 1] = segmentation.ground->points[i].y;
        buf[buf_index_base + 2] = segmentation.ground->points[i].z;
        buf[buf_index_base + 3] = segmentation.ground->points[i].intensity;
        buf[buf_index_base + 4] = 255.0;
    }
    return result;
}

PYBIND11_MODULE(ground_removal_ext, m) {
    m.doc() = "RANSAC Based groud removal";

    m.def("ground_removal_kernel", &ground_removal_kernel, "ground removal",
          py::arg("input"), py::arg("distance_th"), py::arg("iter_times")
    );

}
