#ifndef UTILS
#define UTILS

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <image_geometry/stereo_camera_model.h>

using namespace std;
using namespace cv;

namespace haloc
{

class Utils
{

public:

  /** \brief extract the keypoints of some image
    * @return
    * \param image the source image
    * \param key_points is the pointer for the resulting image key_points
    * \param type descriptor type (see opencv docs)
    */
  static void keypointDetector(const Mat& image,
                               vector<KeyPoint>& key_points,
                               string type)
  {
    // Check Opponent color space descriptors
    size_t pos = 0;
    if ( (pos=type.find("Opponent")) == 0)
    {
      pos += string("Opponent").size();
      type = type.substr(pos);
    }

    initModule_nonfree();
    Ptr<FeatureDetector> cv_detector;
    cv_detector = FeatureDetector::create(type);
    try
    {
      cv_detector->detect(image, key_points);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_detector exception: %s", e.what());
    }
  }

  /** \brief extract descriptors of some image
    * @return
    * \param image the source image
    * \param key_points keypoints of the source image
    * \param descriptors is the pointer for the resulting image descriptors
    */
  static void descriptorExtraction(const Mat& image,
   vector<KeyPoint>& key_points, Mat& descriptors, string type)
  {
    Ptr<DescriptorExtractor> cv_extractor;
    cv_extractor = DescriptorExtractor::create(type);
    try
    {
      cv_extractor->compute(image, key_points, descriptors);
    }
    catch (Exception& e)
    {
      ROS_WARN("[StereoSlam:] cv_extractor exception: %s", e.what());
    }
  }

  /** \brief match descriptors of 2 images by ratio
    * @return
    * \param descriptors1 descriptors of image1
    * \param descriptors2 descriptors of image2
    * \param ratio to determine correct matchings
    * \param matches output vector with the matches
    */
  static void ratioMatching(const Mat& descriptors1, const Mat& descriptors2,
    double ratio, const Mat& match_mask, vector<DMatch>& matches)
  {
    matches.clear();
    if (descriptors1.empty() || descriptors2.empty())
      return;
    assert(descriptors1.type() == descriptors2.type());
    assert(descriptors1.cols == descriptors2.cols);

    const int knn = 2;
    Ptr<DescriptorMatcher> descriptor_matcher;
    // choose matcher based on feature type
    if (descriptors1.type() == CV_8U)
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce");
    }
    vector<vector<DMatch> > knn_matches;
    descriptor_matcher->knnMatch(descriptors1, descriptors2,
            knn_matches, knn, match_mask);

    for (unsigned m = 0; m < knn_matches.size(); m++)
    {
      if (knn_matches[m].size() < 2) continue;
      if (knn_matches[m][0].distance <= knn_matches[m][1].distance * ratio)
        matches.push_back(knn_matches[m][0]);
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return
    * \param descriptors1 descriptors of image1
    * \param descriptors2 descriptors of image2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void thresholdMatching(const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask, vector<DMatch>& matches)
  {
    matches.clear();
    if (descriptors1.empty() || descriptors2.empty())
      return;
    assert(descriptors1.type() == descriptors2.type());
    assert(descriptors1.cols == descriptors2.cols);

    const int knn = 2;
    Ptr<DescriptorMatcher> descriptor_matcher;
    // choose matcher based on feature type
    if (descriptors1.type() == CV_8U)
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce-Hamming");
    }
    else
    {
      descriptor_matcher = DescriptorMatcher::create("BruteForce");
    }
    vector<vector<DMatch> > knn_matches;
    descriptor_matcher->knnMatch(descriptors1, descriptors2,
            knn_matches, knn, match_mask);

    for (size_t m = 0; m < knn_matches.size(); m++)
    {
      if (knn_matches[m].size() < 2) continue;
      float dist1 = knn_matches[m][0].distance;
      float dist2 = knn_matches[m][1].distance;
      if (dist1 / dist2 < threshold)
      {
        matches.push_back(knn_matches[m][0]);
      }
    }
  }

  /** \brief filter matches of cross check matching
    * @return
    * \param matches1to2 matches from image 1 to 2
    * \param matches2to1 matches from image 2 to 1
    * \param matches output vector with filtered matches
    */
  static void crossCheckFilter(
      const vector<DMatch>& matches1to2,
      const vector<DMatch>& matches2to1,
      vector<DMatch>& checked_matches)
  {
    checked_matches.clear();
    for (size_t i = 0; i < matches1to2.size(); ++i)
    {
      bool match_found = false;
      const DMatch& forward_match = matches1to2[i];
      for (size_t j = 0; j < matches2to1.size() && match_found == false; ++j)
      {
        const DMatch& backward_match = matches2to1[j];
        if (forward_match.trainIdx == backward_match.queryIdx &&
            forward_match.queryIdx == backward_match.trainIdx)
        {
          checked_matches.push_back(forward_match);
          match_found = true;
        }
      }
    }
  }

  /** \brief match descriptors of 2 images by threshold
    * @return
    * \param descriptors1 descriptors of image 1
    * \param descriptors2 descriptors of image 2
    * \param threshold to determine correct matchings
    * \param match_mask mask for matchings
    * \param matches output vector with the matches
    */
  static void crossCheckThresholdMatching(
    const Mat& descriptors1, const Mat& descriptors2,
    double threshold, const Mat& match_mask,
    vector<DMatch>& matches)
  {
    vector<DMatch> query_to_train_matches;
    thresholdMatching(descriptors1, descriptors2, threshold, match_mask, query_to_train_matches);
    vector<DMatch> train_to_query_matches;
    Mat match_mask_t;
    if (!match_mask.empty()) match_mask_t = match_mask.t();
    thresholdMatching(descriptors2, descriptors1, threshold, match_mask_t, train_to_query_matches);

    crossCheckFilter(query_to_train_matches, train_to_query_matches, matches);
  }

  /** \brief Compute the 3D point projecting the disparity
    * @return
    * \param stereo_camera_model is the camera model
    * \param left_point on the left image
    * \param right_point on the right image
    * \param world_point pointer to the corresponding 3d point
    */
  static bool calculate3DPoint(const image_geometry::StereoCameraModel stereo_camera_model,
                               const Point2d& left_point,
                               const Point2d& right_point,
                               double max_proj_err,
                               Point3d& world_point)
  {
    // Get the world point
    double disparity = left_point.x - right_point.x;
    stereo_camera_model.projectDisparityTo3d(left_point, disparity, world_point);

    // Filter the given world point
    // Extract camera parameters
    double fx_l = stereo_camera_model.left().fx();
    double fy_l = stereo_camera_model.left().fy();
    double cx_l = stereo_camera_model.left().cx();
    double cy_l = stereo_camera_model.left().cy();

    double fx_r = stereo_camera_model.right().fx();
    double fy_r = stereo_camera_model.right().fy();
    double cx_r = stereo_camera_model.right().cx();
    double cy_r = stereo_camera_model.right().cy();

    double baseline = stereo_camera_model.baseline();

    // Direccion vectors
    double x_l = (1.0/fx_l) * (left_point.x - cx_l);
    double y_l = (1.0/fy_l) * (left_point.y - cy_l);
    double z_l = 1.0;

    double x_r = (1.0/fx_r) * (right_point.x - cx_r);
    double y_r = (1.0/fy_r) * (right_point.y - cy_r);
    double z_r = 1.0;

    tf::Vector3 u(x_l, y_l, z_l);
    tf::Vector3 v(x_r, y_r, z_r);
    tf::Vector3 w0(-baseline, 0.0, 0.0);

    // Origins
    tf::Vector3 l0(0.0, 0.0, 0.0);
    tf::Vector3 r0(baseline, 0.0, 0.0);

    // Temporal variables
    double a = u.dot(u);
    double b = u.dot(v);
    double c = v.dot(v);
    double d = u.dot(w0);
    double e = v.dot(w0);

    // Minimum distance between the two projected points
    tf::Vector3 dv = (l0-r0) + ( ( (b*e-c*d)*u - (a*e-b*d)*v ) / (a*c - b*b) );
    double dist = sqrt( dv.x()*dv.x() + dv.y()*dv.y() + dv.z()*dv.z() );
    if (dist > max_proj_err)
    {
      return false;
    }
    else
    {
      return true;
    }
  }

  /** \brief Sort 2 matchings by value
    * @return true if matching 1 is smaller than matching 2
    * \param matching 1
    * \param matching 2
    */
  static bool sortByMatching(const pair<int, float> d1, const pair<int, float> d2)
  {
    return (d1.second < d2.second);
  }

  /** \brief Sort 2 descriptors matchings by distance
    * @return true if vector 1 is smaller than vector 2
    * \param descriptor matching 1
    * \param descriptor matching 2
    */
  static bool sortDescByDistance(const DMatch& d1, const DMatch& d2)
  {
    return (d1.distance < d2.distance);
  }

  /** \brief Sort 2 keypoints by its response
    * @return true if response of kp 1 is greater than response of kp 2
    * \param likelihood 1
    * \param likelihood 2
    */
  static bool sortByResponse(const KeyPoint kp1, const KeyPoint kp2)
  {
    return (kp1.response > kp2.response);
  }

  /** \brief Sort 2 matchings by likelihood
    * @return true if likelihood 1 is greater than likelihood 2
    * \param likelihood 1
    * \param likelihood 2
    */
  static bool sortByLikelihood(const pair<int, float> p1, const pair<int, float> p2)
  {
    return (p1.second > p2.second);
  }

  /** \brief compose the transformation matrix using 2 Mat as inputs:
    * one for rotation and one for translation
    * @return the transformation matrix
    * \param rvec cv matrix with the rotation angles
    * \param tvec cv matrix with the transformation x y z
    */
  static tf::Transform buildTransformation(Mat rvec, Mat tvec)
  {
    if (rvec.empty() || tvec.empty())
      return tf::Transform();

    tf::Vector3 axis(rvec.at<double>(0, 0),
               rvec.at<double>(1, 0),
                 rvec.at<double>(2, 0));
    double angle = norm(rvec);
    tf::Quaternion quaternion(axis, angle);

    tf::Vector3 translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
        tvec.at<double>(2, 0));

    return tf::Transform(quaternion, translation);
  }
};

} // namespace

#endif // UTILS


