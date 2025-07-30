#include "ArucoTracker.hpp"
#include <sstream>

ArucoTrackerNode::ArucoTrackerNode()
	: Node("aruco_tracker_node")
{
	loadParameters();

	// Create detectors for both marker types
	auto outer_dictionary = cv::aruco::getPredefinedDictionary(_param_outer_dictionary);
	auto inner_dictionary = cv::aruco::getPredefinedDictionary(_param_inner_dictionary);

	_outer_detector = std::make_unique<cv::aruco::ArucoDetector>(outer_dictionary, cv::aruco::DetectorParameters());
	_inner_detector = std::make_unique<cv::aruco::ArucoDetector>(inner_dictionary, cv::aruco::DetectorParameters());

	auto qos = rclcpp::QoS(1).best_effort();

	_image_sub = create_subscription<sensor_msgs::msg::Image>(
			     "/camera", qos, std::bind(&ArucoTrackerNode::image_callback, this, std::placeholders::_1));

	_camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
				   "/camera_info", qos, std::bind(&ArucoTrackerNode::camera_info_callback, this, std::placeholders::_1));

	// Publishers
	_image_pub = create_publisher<sensor_msgs::msg::Image>("/image_proc", qos);
	_target_pose_pub = create_publisher<geometry_msgs::msg::PoseStamped>("/target_pose", qos);
}

void ArucoTrackerNode::loadParameters()
{
	// Outer marker parameters (7x7)
	declare_parameter<int>("outer_aruco_id", 442);
	declare_parameter<int>("outer_dictionary", 15); // DICT_7X7_1000
	declare_parameter<double>("outer_marker_size", 0.3048);

	// Inner marker parameters (7x7)
	declare_parameter<int>("inner_aruco_id", 11);  // Blackest ArUco ID for better detection
	declare_parameter<int>("inner_dictionary", 15); // DICT_7X7_1000
	declare_parameter<double>("inner_marker_size", 0.0339);

	// Detection priority
	declare_parameter<bool>("prefer_inner_marker", true);

	get_parameter("outer_aruco_id", _param_outer_aruco_id);
	get_parameter("outer_dictionary", _param_outer_dictionary);
	get_parameter("outer_marker_size", _param_outer_marker_size);
	get_parameter("inner_aruco_id", _param_inner_aruco_id);
	get_parameter("inner_dictionary", _param_inner_dictionary);
	get_parameter("inner_marker_size", _param_inner_marker_size);
	get_parameter("prefer_inner_marker", _param_prefer_inner_marker);
}

void ArucoTrackerNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	try {
		// Convert ROS image message to OpenCV image
		cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

		// Detect markers with both detectors
		std::vector<int> outer_ids, inner_ids;
		std::vector<std::vector<cv::Point2f>> outer_corners, inner_corners;
		
		_outer_detector->detectMarkers(cv_ptr->image, outer_corners, outer_ids);
		_inner_detector->detectMarkers(cv_ptr->image, inner_corners, inner_ids);

		// Debug: Log detected markers
		if (!outer_ids.empty()) {
			RCLCPP_INFO(get_logger(), "Detected outer markers: %s", 
				[&outer_ids]() { std::string s; for (int id : outer_ids) s += std::to_string(id) + " "; return s; }().c_str());
		}
		if (!inner_ids.empty()) {
			RCLCPP_INFO(get_logger(), "Detected inner markers: %s", 
				[&inner_ids]() { std::string s; for (int id : inner_ids) s += std::to_string(id) + " "; return s; }().c_str());
		}

		// Draw all detected markers
		cv::aruco::drawDetectedMarkers(cv_ptr->image, outer_corners, outer_ids);
		cv::aruco::drawDetectedMarkers(cv_ptr->image, inner_corners, inner_ids);

		if (!_camera_matrix.empty() && !_dist_coeffs.empty()) {

			// Process inner markers first if preferred
			if (_param_prefer_inner_marker) {
				for (size_t i = 0; i < inner_ids.size(); i++) {
					if (inner_ids[i] == _param_inner_aruco_id) {
						// Process inner marker
						std::vector<cv::Point2f> undistorted_corners;
						cv::undistortPoints(inner_corners[i], undistorted_corners, _camera_matrix, cv::noArray(), cv::noArray(), _camera_matrix);

						float half_size = _param_inner_marker_size / 2.0f;
						std::vector<cv::Point3f> objectPoints = {
							cv::Point3f(-half_size,  half_size, 0),
							cv::Point3f(half_size,  half_size, 0),
							cv::Point3f(half_size, -half_size, 0),
							cv::Point3f(-half_size, -half_size, 0)
						};

						cv::Vec3d rvec, tvec;
						cv::solvePnP(objectPoints, undistorted_corners, _camera_matrix, cv::noArray(), rvec, tvec);
						cv::drawFrameAxes(cv_ptr->image, _camera_matrix, cv::noArray(), rvec, tvec, _param_inner_marker_size);

						// Publish pose
						cv::Mat rot_mat;
						cv::Rodrigues(rvec, rot_mat);
						cv::Quatd quat = cv::Quatd::createFromRotMat(rot_mat).normalize();

						geometry_msgs::msg::PoseStamped pose_msg;
						pose_msg.header.stamp = msg->header.stamp;
						pose_msg.header.frame_id = "camera_frame";
						pose_msg.pose.position.x = tvec[0];
						pose_msg.pose.position.y = tvec[1];
						pose_msg.pose.position.z = tvec[2];
						pose_msg.pose.orientation.x = quat.x;
						pose_msg.pose.orientation.y = quat.y;
						pose_msg.pose.orientation.z = quat.z;
						pose_msg.pose.orientation.w = quat.w;

						_target_pose_pub->publish(pose_msg);
						annotate_image(cv_ptr, tvec);
						return; // Exit after processing inner marker
					}
				}
			}

			// Process outer markers if no inner marker found or not preferred
			for (size_t i = 0; i < outer_ids.size(); i++) {
				if (outer_ids[i] == _param_outer_aruco_id) {
					// Process outer marker
					std::vector<cv::Point2f> undistorted_corners;
					cv::undistortPoints(outer_corners[i], undistorted_corners, _camera_matrix, cv::noArray(), cv::noArray(), _camera_matrix);

					float half_size = _param_outer_marker_size / 2.0f;
					std::vector<cv::Point3f> objectPoints = {
						cv::Point3f(-half_size,  half_size, 0),
						cv::Point3f(half_size,  half_size, 0),
						cv::Point3f(half_size, -half_size, 0),
						cv::Point3f(-half_size, -half_size, 0)
					};

					cv::Vec3d rvec, tvec;
					cv::solvePnP(objectPoints, undistorted_corners, _camera_matrix, cv::noArray(), rvec, tvec);
					cv::drawFrameAxes(cv_ptr->image, _camera_matrix, cv::noArray(), rvec, tvec, _param_outer_marker_size);

					// Publish pose
					cv::Mat rot_mat;
					cv::Rodrigues(rvec, rot_mat);
					cv::Quatd quat = cv::Quatd::createFromRotMat(rot_mat).normalize();

					geometry_msgs::msg::PoseStamped pose_msg;
					pose_msg.header.stamp = msg->header.stamp;
					pose_msg.header.frame_id = "camera_frame";
					pose_msg.pose.position.x = tvec[0];
					pose_msg.pose.position.y = tvec[1];
					pose_msg.pose.position.z = tvec[2];
					pose_msg.pose.orientation.x = quat.x;
					pose_msg.pose.orientation.y = quat.y;
					pose_msg.pose.orientation.z = quat.z;
					pose_msg.pose.orientation.w = quat.w;

					_target_pose_pub->publish(pose_msg);
					annotate_image(cv_ptr, tvec);
					break;
				}
			}

		} else {
			RCLCPP_ERROR(get_logger(), "Missing camera calibration");
		}

		// Always publish image
		cv_bridge::CvImage out_msg;
		out_msg.header = msg->header;
		out_msg.encoding = sensor_msgs::image_encodings::BGR8;
		out_msg.image = cv_ptr->image;
		_image_pub->publish(*out_msg.toImageMsg().get());

	} catch (const cv_bridge::Exception& e) {
		RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
	}
}

void ArucoTrackerNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	// Always update the camera matrix and distortion coefficients from the new message
	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();   // Use clone to ensure a deep copy
	_dist_coeffs = cv::Mat(msg->d.size(), 1, CV_64F, const_cast<double*>(msg->d.data())).clone();   // Use clone to ensure a deep copy

	// Log the first row of the camera matrix to verify correct values
	RCLCPP_INFO(get_logger(), "Camera matrix updated:\n[%f, %f, %f]\n[%f, %f, %f]\n[%f, %f, %f]",
		    _camera_matrix.at<double>(0, 0), _camera_matrix.at<double>(0, 1), _camera_matrix.at<double>(0, 2),
		    _camera_matrix.at<double>(1, 0), _camera_matrix.at<double>(1, 1), _camera_matrix.at<double>(1, 2),
		    _camera_matrix.at<double>(2, 0), _camera_matrix.at<double>(2, 1), _camera_matrix.at<double>(2, 2));
	RCLCPP_INFO(get_logger(), "Camera Matrix: fx=%f, fy=%f, cx=%f, cy=%f",
		    _camera_matrix.at<double>(0, 0), // fx
		    _camera_matrix.at<double>(1, 1), // fy
		    _camera_matrix.at<double>(0, 2), // cx
		    _camera_matrix.at<double>(1, 2)  // cy
		   );

	// Check if focal length is zero after update
	if (_camera_matrix.at<double>(0, 0) == 0) {
		RCLCPP_ERROR(get_logger(), "Focal length is zero after update!");

	} else {
		RCLCPP_INFO(get_logger(), "Updated camera intrinsics from camera_info topic.");

		RCLCPP_INFO(get_logger(), "Unsubscribing from camera info topic");
		_camera_info_sub.reset();
	}
}

void ArucoTrackerNode::annotate_image(cv_bridge::CvImagePtr image, const cv::Vec3d& target)
{
	// Annotate the image with the target position and marker size
	std::ostringstream stream;
	stream << std::fixed << std::setprecision(2);
	stream << "X: "  << target[0] << " Y: " << target[1]  << " Z: " << target[2];
	std::string text_xyz = stream.str();

	int fontFace = cv::FONT_HERSHEY_SIMPLEX;
	double fontScale = 1;
	int thickness = 2;
	int baseline = 0;
	cv::Size textSize = cv::getTextSize(text_xyz, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;
	cv::Point textOrg((image->image.cols - textSize.width - 10), (image->image.rows - 10));
	cv::putText(image->image, text_xyz, textOrg, fontFace, fontScale, cv::Scalar(0, 255, 255), thickness, 8);
}

int main(int argc, char** argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<ArucoTrackerNode>());
	rclcpp::shutdown();
	return 0;
}