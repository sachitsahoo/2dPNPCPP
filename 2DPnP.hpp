#ifndef TWOD_PNP_HPP
#define TWOD_PNP_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <lsqcpp/lsqcpp.hpp>
#include <cmath>

using std::vector;
using std::cout;
using std::endl;
using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixX2d;
using Eigen::MatrixX3d;
using Eigen::MatrixXd;
using Eigen::JacobiSVD;
using Eigen::ComputeFullU;
using Eigen::ComputeFullV;
//stupid amount of using Eigens because namespace overlap :(
using lsqcpp::LevenbergMarquardt;

struct Point2D {
    double x, y;
};

// Define a least-squares cost function for reprojection error
struct ReprojectionError {
    Point2D world_point;
    Point2D image_point;

    ReprojectionError(Point2D wp, Point2D ip);

    template <typename Scalar>
    void operator()(const Eigen::Matrix<Scalar, 3, 1>& camera, Eigen::Matrix<Scalar, 2, 1>& residuals) const;
};

void computeInitialPose(const std::vector<Point2D>& worldPoints, const std::vector<Point2D>& imagePoints, Eigen::Matrix3d& rotation, Eigen::Vector2d& translation);

void refinePose(const std::vector<Point2D>& worldPoints, const std::vector<Point2D>& imagePoints, Eigen::Matrix3d& rotation, Eigen::Vector2d& translation);

#endif // TWOD_PNP_HPP
