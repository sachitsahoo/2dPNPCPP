#include "2DPnP.hpp"
#include "lsqcpp/lsqcpp.hpp"

ReprojectionError::ReprojectionError(Point2D wp, Point2D ip)
    : world_point(wp), image_point(ip) {}

template <typename Scalar>
void ReprojectionError::operator()(const Eigen::Matrix<Scalar, 3, 1>& camera, Eigen::Matrix<Scalar, 2, 1>& residuals) const {
    // camera[0] and camera[1] are the x and y coordinates of the camera center
    // camera[2] is the rotation angle of the camera
    Scalar predicted_x = cos(camera(2)) * Scalar(world_point.x) - sin(camera(2)) * Scalar(world_point.y) + camera(0);
    Scalar predicted_y = sin(camera(2)) * Scalar(world_point.x) + cos(camera(2)) * Scalar(world_point.y) + camera(1);

    residuals(0) = predicted_x - Scalar(image_point.x);
    residuals(1) = predicted_y - Scalar(image_point.y);
}

// Function to compute initial guess for pose
void computeInitialPose(const vector<Point2D>& worldPoints, const vector<Point2D>& imagePoints, Matrix3d& rotation, Vector2d& translation) {
    int n = worldPoints.size();
    
    // Compute centroids of world and image points
    Vector2d worldCentroid(0, 0);
    Vector2d imageCentroid(0, 0);
    for (int i = 0; i < n; ++i) {
        worldCentroid.x() += worldPoints[i].x;
        worldCentroid.y() += worldPoints[i].y;
        imageCentroid.x() += imagePoints[i].x;
        imageCentroid.y() += imagePoints[i].y;
    }
    worldCentroid /= n;
    imageCentroid /= n;

    // Subtract centroids to get zero-centered coordinates
    MatrixXd W(2, n), I(2, n);
    for (int i = 0; i < n; ++i) {
        W(0, i) = worldPoints[i].x - worldCentroid.x();
        W(1, i) = worldPoints[i].y - worldCentroid.y();
        I(0, i) = imagePoints[i].x - imageCentroid.x();
        I(1, i) = imagePoints[i].y - imageCentroid.y();
    }

    // Compute SVD of W * I^T
    Matrix2d covarianceMatrix = W * I.transpose();
    JacobiSVD<Matrix2d> svd(covarianceMatrix, ComputeFullU | ComputeFullV);
    
    // Compute rotation matrix from SVD
    Matrix2d U = svd.matrixU();
    Matrix2d V = svd.matrixV();
    Matrix2d R = V * U.transpose();

    if (R.determinant() < 0) {
        V.col(1) *= -1;
        R = V * U.transpose();
    }

    // Set the 3x3 rotation matrix (embedding 2D rotation into 3D space)
    rotation = Matrix3d::Identity();
    rotation.block<2, 2>(0, 0) = R;

    // Compute translation: t = centroid_image - R * centroid_world
    translation = imageCentroid - R * worldCentroid;
}

// Function to refine the pose using least-squares optimization (lsqcpp)
void refinePose(const vector<Point2D>& worldPoints, const vector<Point2D>& imagePoints, Matrix3d& rotation, Vector2d& translation) {
    Eigen::Vector3d camera;
    camera << translation.x(), translation.y(), atan2(rotation(1, 0), rotation(0, 0));

    // Define the functor that lsqcpp will optimize
    auto objective = [&worldPoints, &imagePoints](const Eigen::Vector3d& camera, Eigen::VectorXd& residuals) {
        residuals.resize(2 * worldPoints.size());
        for (size_t i = 0; i < worldPoints.size(); ++i) {
            ReprojectionError error(worldPoints[i], imagePoints[i]);
            Eigen::Vector2d res;
            error(camera, res);
            residuals.segment<2>(2 * i) = res;
        }
    };

    // Setup the Levenberg-Marquardt optimizer
    lsqcpp::LevenbergMarquardtX<double, decltype(objective)> optimizer;
    optimizer.setObjective(objective);
    optimizer.setMaximumIterations(100);
    optimizer.setNumericalEpsilon(1e-6);

    // Solve the optimization problem
    auto result = optimizer.minimize(camera);

    // Update the pose based on optimization results
    translation.x() = camera(0);
    translation.y() = camera(1);
    rotation = Matrix3d::Identity();
    rotation(0, 0) = cos(camera(2));
    rotation(0, 1) = -sin(camera(2));
    rotation(1, 0) = sin(camera(2));
    rotation(1, 1) = cos(camera(2));
}

int main() {
    // Example world points (in 2D)
    vector<Point2D> worldPoints = { {0, 0}, {1, 0}, {0, 1}, {1, 1} };
    
    // Corresponding image points (observed in the camera)
    vector<Point2D> imagePoints = { {100, 100}, {150, 100}, {100, 150}, {150, 150} };

    // Variables to hold the camera pose (rotation and translation)
    Matrix3d rotation;
    Vector2d translation;

    // Compute initial guess for the pose
    computeInitialPose(worldPoints, imagePoints, rotation, translation);

    // Refine the pose using optimization
    refinePose(worldPoints, imagePoints, rotation, translation);

    // Output the final pose
    cout << "Final Rotation Matrix:\n" << rotation << endl;
    cout << "Final Translation Vector:\n" << translation.transpose() << endl;

    return 0;
}
