#include "libhaloc/image.h"
#include "libhaloc/utils.h"

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
haloc::Image::Params::Params() :
  desc_type("SIFT"),
  desc_matching_type("CROSSCHECK"),
  desc_thresh_ratio(DEFAULT_DESC_THRESH_RATIO),
  min_matches(DEFAULT_MIN_MATCHES),
  epipolar_thresh(DEFAULT_EPIPOLAR_THRESH),
  b_width(DEFAULT_B_WIDTH),
  b_height(DEFAULT_B_HEIGHT),
  b_max_features(DEFAULT_B_MAX_FEATURES)
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
int haloc::Image::getId() { return id_; }
void haloc::Image::setId(int id) { id_ = id; }
vector<KeyPoint> haloc::Image::getKp() { return kp_; }
void haloc::Image::setKp(vector<KeyPoint> kp) { kp_ = kp; }
Mat haloc::Image::getDesc() { return desc_; }
void haloc::Image::setDesc(Mat desc) { desc_ = desc; }
vector<Point3f> haloc::Image::get3D() { return points_3d_; };
void haloc::Image::set3D(vector<Point3f> points_3d) { points_3d_ = points_3d; };


/** \brief Sets the stereo camera model for the class
  * \param stereo_camera_model.
  */
void haloc::Image::setCameraModel(image_geometry::StereoCameraModel stereo_camera_model)
{
  stereo_camera_model_ = stereo_camera_model;
}

/** \brief Compute the keypoints and descriptors for one image (mono)
  * @return false if not enough keypoints. True otherwise.
  * \param Image identifier
  * \param cvMat image
  */
bool haloc::Image::setMono(int id, const Mat& img)
{
  // Reset
  kp_.clear();
  desc_.release();
  id_ = id;

  // Equalize image
  Mat img_gs;
  cvtColor(img, img_gs, CV_BGR2GRAY);
  equalizeHist(img_gs, img_gs);

  // Extract keypoints
  vector<KeyPoint> kp;
  haloc::Utils::keypointDetector(img_gs, kp, params_.desc_type);

  // Check if the number of kp is enough for the computation of the 3D
  if (kp.size() < params_.min_matches)
    return false;

  // Extract descriptors
  desc_.release();
  haloc::Utils::descriptorExtraction(img_gs, kp, desc_, params_.desc_type);

  // Convert kp
  kp_.clear();
  kp_ = kp;

  return true;
}

/** \brief Compute the keypoints and descriptors for two images (stereo)
  * @return false if not enough stereo matches. True otherwise.
  * \param Image identifier
  * \param cvMat image for left frame
  * \param cvMat image for right frame
  */
bool haloc::Image::setStereo(int id, const Mat& img_l, const Mat& img_r)
{
  // Reset
  kp_.clear();
  desc_.release();
  points_3d_.clear();
  id_ = id;

  // Equalize image
  Mat img_l_gs, img_r_gs;
  cvtColor(img_l, img_l_gs, CV_BGR2GRAY);
  cvtColor(img_r, img_r_gs, CV_BGR2GRAY);
  equalizeHist(img_l_gs, img_l_gs);
  equalizeHist(img_r_gs, img_r_gs);

  // Extract keypoints (left)
  vector<KeyPoint> kp_l;
  haloc::Utils::keypointDetector(img_l_gs, kp_l, params_.desc_type);

  // Extract descriptors (left)
  Mat desc_l;
  haloc::Utils::descriptorExtraction(img_l_gs, kp_l, desc_l, params_.desc_type);

  // Extract keypoints (right)
  vector<KeyPoint> kp_r;
  haloc::Utils::keypointDetector(img_r_gs, kp_r, params_.desc_type);

  // Extract descriptors (right)
  Mat desc_r;
  haloc::Utils::descriptorExtraction(img_r_gs, kp_r, desc_r, params_.desc_type);

  // Find matches between left and right images
  Mat match_mask;
  vector<DMatch> matches, matches_filtered;

  if(params_.desc_matching_type == "CROSSCHECK")
  {
    haloc::Utils::crossCheckThresholdMatching(desc_l,
        desc_r, params_.desc_thresh_ratio, match_mask, matches);
  }
  else if (params_.desc_matching_type == "RATIO")
  {
    haloc::Utils::ratioMatching(desc_l,
        desc_r, params_.desc_thresh_ratio, match_mask, matches);
  }
  else
  {
    ROS_ERROR("[Haloc:] ERROR -> desc_matching_type must be 'CROSSCHECK' or 'RATIO'");
  }

  // Filter matches by epipolar
  for (size_t i = 0; i < matches.size(); ++i)
  {
    if (abs(kp_l[matches[i].queryIdx].pt.y - kp_r[matches[i].trainIdx].pt.y)
        < params_.epipolar_thresh)
      matches_filtered.push_back(matches[i]);
  }

  // Bucket features
  matches_filtered = bucketFeatures(matches_filtered, kp_l);

  // Check if the number of matches is enough for the computation of the 3D
  if (matches_filtered.size() < params_.min_matches)
    return false;

  // Compute 3D points
  vector<KeyPoint> matched_kp_l;
  vector<Point3f> matched_3d_points;
  Mat matched_desc_l;
  for (size_t i=0; i<matches_filtered.size(); ++i)
  {
    int index_left = matches_filtered[i].queryIdx;
    int index_right = matches_filtered[i].trainIdx;
    Point3d world_point;
    bool valid = haloc::Utils::calculate3DPoint(stereo_camera_model_,
                                                kp_l[index_left].pt,
                                                kp_r[index_right].pt,
                                                0.002,
                                                world_point);

    if (valid)
    {
      matched_kp_l.push_back(kp_l[index_left]);
      matched_desc_l.push_back(desc_l.row(index_left));
      matched_3d_points.push_back(world_point);
    }
  }

  // Save descriptors
  desc_ = matched_desc_l;

  // Convert keypoints
  kp_.clear();
  kp_ = matched_kp_l;

  // Save 3D
  points_3d_ = matched_3d_points;

  return true;
}

/** \brief Bucket features
  * @return the bucketed matches
  * \param input matches
  * \param input keypoints
  * \param bucket with
  * \param bucket height
  * \param number of features per bucket
  */
vector<DMatch> haloc::Image::bucketFeatures(vector<DMatch> matches,
                                            vector<KeyPoint> kp)
{
  // Find max values
  float x_max = 0;
  float y_max = 0;
  for (vector<DMatch>::iterator it = matches.begin(); it!=matches.end(); it++)
  {
    if (kp[it->queryIdx].pt.x > x_max) x_max = kp[it->queryIdx].pt.x;
    if (kp[it->queryIdx].pt.y > y_max) y_max = kp[it->queryIdx].pt.y;
  }

  // Allocate number of buckets needed
  int bucket_cols = (int)floor(x_max/params_.b_width) + 1;
  int bucket_rows = (int)floor(y_max/params_.b_height) + 1;
  vector<DMatch> *buckets = new vector<DMatch>[bucket_cols*bucket_rows];

  // Assign matches to their buckets
  for (vector<DMatch>::iterator it=matches.begin(); it!=matches.end(); it++)
  {
    int u = (int)floor(kp[it->queryIdx].pt.x/params_.b_width);
    int v = (int)floor(kp[it->queryIdx].pt.y/params_.b_height);
    buckets[v*bucket_cols+u].push_back(*it);
  }

  // Refill matches from buckets
  vector<DMatch> output;
  for (int i=0; i<bucket_cols*bucket_rows; i++)
  {
    // Sort descriptors matched by distance
    sort(buckets[i].begin(), buckets[i].end(), haloc::Utils::sortDescByDistance);

    // Add up to max_features features from this bucket to output
    int k=0;
    for (vector<cv::DMatch>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++)
    {
      output.push_back(*it);
      k++;
      if (k >= params_.b_max_features)
        break;
    }
  }
  return output;
}