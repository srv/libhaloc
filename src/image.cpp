#include "libhaloc/image.h"
#include "libhaloc/utils.h"

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
haloc::Image::Params::Params() :
  desc_type("SIFT"),
  desc_thresh(DEFAULT_DESC_THRESH),
  epipolar_thresh(DEFAULT_EPIPOLAR_THRESH)
{}

/** \brief Image constructor
  */
haloc::Image::Image() {}

/** \brief Sets the parameters
  * \param parameter struct.
  */
void haloc::Image::setParams(const Params& params) 
{
  params_ = params;
}

// Access specifiers
vector<Point2f> haloc::Image::getKp() { return kp_; }
Mat haloc::Image::getDesc() { return desc_; }
vector<Point3f> haloc::Image::get3D() { return points_3d_; };


/** \brief Sets the stereo camera model for the class
  * \param stereo_camera_model.
  */
void haloc::Image::setCameraModel(image_geometry::StereoCameraModel stereo_camera_model)
{
  stereo_camera_model_ = stereo_camera_model;
}

/** \brief Compute the keypoints and descriptors for one image (mono)
  * \param cvMat image
  */
void haloc::Image::setMono(const Mat& img)
{
  // Extract keypoints
  kp_.clear();
  desc_.release();
  vector<KeyPoint> kp;
  haloc::Utils::keypointDetector(img, kp, params_.desc_type);

  // Extract descriptors
  haloc::Utils::descriptorExtraction(img, kp, desc_, params_.desc_type);

  // Convert
  for(int i=0; i<kp.size(); i++)
    kp_.push_back(kp[i].pt);
}

/** \brief Compute the keypoints and descriptors for two images (stereo)
  * \param cvMat image for left frame
  * \param cvMat image for right frame
  */
void haloc::Image::setStereo(const Mat& img_l, const Mat& img_r)
{
  // Extract keypoints (left)
  kp_.clear();
  desc_.release();
  vector<KeyPoint> kp_l;
  haloc::Utils::keypointDetector(img_l, kp_l, params_.desc_type);

  // Extract descriptors (left)
  haloc::Utils::descriptorExtraction(img_l, kp_l, desc_, params_.desc_type);

  // Extract keypoints (right)
  Mat desc_r;
  vector<KeyPoint> kp_r;
  haloc::Utils::keypointDetector(img_r, kp_r, params_.desc_type);

  // Extract descriptors (right)
  haloc::Utils::descriptorExtraction(img_r, kp_r, desc_r, params_.desc_type);

  // Find matches between left and right images
  Mat match_mask;
  vector<DMatch> matches, matches_filtered;
  haloc::Utils::crossCheckThresholdMatching(desc_, 
      desc_r, params_.desc_thresh, match_mask, matches);

  // Filter matches by epipolar 
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (abs(kp_l[matches[i].queryIdx].pt.y - kp_r[matches[i].trainIdx].pt.y) 
        < params_.epipolar_thresh)
      matches_filtered.push_back(matches[i]);
  }

  // Compute 3D points
  vector<KeyPoint> matched_kp_l;
  vector<Point3f> matched_3d_points;
  Mat matched_desc_l;
  for (size_t i = 0; i < matches_filtered.size(); ++i)
  {
    int index_left = matches_filtered[i].queryIdx;
    int index_right = matches_filtered[i].trainIdx;
    Point3d world_point;
    haloc::Utils::calculate3DPoint( stereo_camera_model_,
                                    kp_l[index_left].pt,
                                    kp_r[index_right].pt,
                                    world_point);
    matched_kp_l.push_back(kp_l[index_left]);
    matched_desc_l.push_back(desc_.row(index_left));
    matched_3d_points.push_back(world_point);
  }

  // Save properties
  for(int i=0; i<matched_kp_l.size(); i++)
    kp_.push_back(matched_kp_l[i].pt);
  desc_ = matched_desc_l;
  points_3d_ = matched_3d_points;
}