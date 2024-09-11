#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <eigen/Dense>
#include <math.h>
#include "2DPnP.hpp"

using Eigen::Matrix3d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Matrix;
using Eigen::Matrix2d;
// Simple Point2D structure to hold 2D points
struct Point2D {
    double x;
    double y;
};

// Helper function to generate random world points in a specified range
std::vector<Point2D> generateWorldPoints(int numPoints) {
    std::vector<Point2D> points;
    for (int i = 0; i < numPoints; ++i) {
        Point2D point = { ((rand() % 2000) / 1000.0) - 1.0, ((rand() % 2000) / 1000.0) - 1.0 };
        points.push_back(point);
    }
    return points;
}

// Helper function to simulate projection into 2D image plane (with noise)
std::vector<Point2D> projectPointsWithNoise(const std::vector<Point2D>& worldPoints, const Eigen::Matrix3d& rotation, const Eigen::Vector2d& translation, double noiseStddev) {
    std::vector<Point2D> imagePoints;
    for (const auto& point : worldPoints) {
        double noisy_x = rotation(0, 0) * point.x + rotation(0, 1) * point.y + translation.x() + ((rand() % 100) / 50.0 - 1) * noiseStddev;
        double noisy_y = rotation(1, 0) * point.x + rotation(1, 1) * point.y + translation.y() + ((rand() % 100) / 50.0 - 1) * noiseStddev;
        imagePoints.push_back({ noisy_x, noisy_y });
    }
    return imagePoints;
}

int main() {
    // Define number of points for testing and noise level
    int numPoints = 50;
    double noiseLevel = 0.02;

    // Generate random world points
    const std::vector<Point2D> worldPoints = generateWorldPoints(numPoints);

    // Define an initial guess for the camera pose (ground truth rotation and translation)
    Eigen::Matrix3d trueRotation;
    Eigen::Vector2d trueTranslation(0.5, -0.3);

    trueRotation = Eigen::Matrix3d::Identity();
    trueRotation(0, 0) = cos(M_PI / 6);  // Example: 30 degrees
    trueRotation(0, 1) = -sin(M_PI / 6);
    trueRotation(1, 0) = sin(M_PI / 6);
    trueRotation(1, 1) = cos(M_PI / 6);

    // Simulate projection to image points with added noise
    const std::vector<Point2D> imagePoints = projectPointsWithNoise(worldPoints, trueRotation, trueTranslation, noiseLevel);

    // Initial guess for rotation and translation
    Eigen::Matrix3d estimatedRotation;
    Eigen::Vector2d estimatedTranslation;

    // Compute the initial guess for the pose using the 2DPnP method
    computeInitialPose(worldPoints, imagePoints, estimatedRotation, estimatedTranslation);

    std::cout << "Initial guess for rotation matrix:\n" << estimatedRotation << std::endl;
    std::cout << "Initial guess for translation vector:\n" << estimatedTranslation.transpose() << std::endl;

    // Refine the estimated pose using the least-squares optimization
    refinePose(worldPoints, imagePoints, estimatedRotation, estimatedTranslation);

    std::cout << "Refined rotation matrix:\n" << estimatedRotation << std::endl;
    std::cout << "Refined translation vector:\n" << estimatedTranslation.transpose() << std::endl;

    return 0;
}
