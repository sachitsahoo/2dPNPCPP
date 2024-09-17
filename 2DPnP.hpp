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

const static double camera_height = 4.5;
//set this to the height of the center of your camera

//Beta = CameraAngle in radians
const static double cameraAngle = 0.0;



// Define a least-squares cost function for reprojection error
struct ReprojectionError {
    Point2D world_point;
    Point2D image_point;

    ReprojectionError(Point2D wp, Point2D ip);

    template <typename Scalar>
    void operator()(const Eigen::Matrix<Scalar, 3, 1>& camera, Eigen::Matrix<Scalar, 2, 1>& residuals) const;
};

struct ReprojectionObjective {
    static constexpr bool ComputesJacobian = true;

    std::vector<Point2D> worldPoints;
    std::vector<Point2D> imagePoints;
    double beta; // Added beta as a member variable

    // ReprojectionObjective(const std::vector<Point2D>& wp, const std::vector<Point2D>& ip, double beta)
    //     : worldPoints(wp), imagePoints(ip), beta(beta) {}

    template<typename Scalar, int Inputs, int Outputs>
    void operator()(const Eigen::Matrix<Scalar, Inputs, 1>& camera, 
                    Eigen::Matrix<Scalar, Outputs, 1>& residuals, 
                    Eigen::Matrix<Scalar, Outputs, Inputs>& jacobian) const {
        
        // Ensure the residuals vector and Jacobian matrix have the correct size
        residuals.resize(2 * worldPoints.size());
        jacobian.setZero(2 * worldPoints.size(), 3); // 3 parameters: [tx, ty, theta]

        Scalar ct = cos(camera(2)); // theta
        Scalar st = sin(camera(2));
        Scalar sb = sin(beta);
        Scalar cb = cos(beta);

        for (size_t i = 0; i < worldPoints.size(); ++i) {
            // Extract world points
            Scalar xi = camera(0) - Scalar(worldPoints[i].x); // camera x - world point x
            Scalar yi = camera(1) - Scalar(worldPoints[i].y); // camera y - world point y
            Scalar zi = camera_height; // Assuming a fixed camera height

            // Compute terms for reprojection error
            Scalar s1 = xi * ct + yi * st;
            Scalar s2 = yi * ct - xi * st;
            Scalar denom = s1 * sb - zi * cb;

            // Compute the residuals (reprojection errors)
            residuals(2 * i) = imagePoints[i].x - (s1 * cb + zi * sb) / denom;
            residuals(2 * i + 1) = imagePoints[i].y - s2 / denom;

            // Compute the Jacobian components
            jacobian(2 * i, 0) = zi * ct / (denom * denom);
            jacobian(2 * i, 1) = zi * st / (denom * denom);
            jacobian(2 * i, 2) = zi * s2 / (denom * denom);

            jacobian(2 * i + 1, 0) = (yi * sb - zi * cb * ct) / (denom * denom);
            jacobian(2 * i + 1, 1) = (zi * cb * ct - xi * sb) / (denom * denom);
            jacobian(2 * i + 1, 2) = (sb * (xi * xi + yi * yi)) / (denom * denom);
        }
    }
};


void computeInitialPose(const std::vector<Point2D>& worldPoints, const std::vector<Point2D>& imagePoints, Eigen::Matrix3d& rotation, Eigen::Vector2d& translation);

void refinePose(const std::vector<Point2D>& worldPoints, const std::vector<Point2D>& imagePoints, Eigen::Matrix3d& rotation, Eigen::Vector2d& translation);

#endif // TWOD_PNP_HPP
