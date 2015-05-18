#ifndef IMAGE_H
#define IMAGE_H

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <image_geometry/stereo_camera_model.h>
#include "utils.h"

using namespace std;
using namespace cv;

namespace haloc
{

class Image
{

public:

  // Class constructor
  Image();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    string desc_type;                   //!> Type of the descriptors (can be SIFT, SURF).
    string desc_matching_type;          //!> Can be "CROSSCHECK" or "RATIO".
    double desc_thresh_ratio;           //!> Descriptor threshold for crosscheck matching (typically between 0.7-0.9) or ratio for ratio matching (typically between 0.6-0.8).
    int min_matches;                    //!> Minimum number of descriptor matches (as for lc.h) here is only used to determine the descriptor matching procedure.
    int epipolar_thresh;                //!> Epipolar threshold (stereo only).
    int b_width;                        //!> Bucket with (only stereo)
    int b_height;                       //!> Bucket height (only stereo)
    int b_max_features;                 //!> Maximum number of features per bucket (only stereo).

    // Default values
    static const double                 DEFAULT_DESC_THRESH_RATIO = 0.8;
    static const int                    DEFAULT_MIN_MATCHES = 20;
    static const int                    DEFAULT_EPIPOLAR_THRESH = 1;
    static const int                    DEFAULT_B_WIDTH = 40;
    static const int                    DEFAULT_B_HEIGHT = 40;
    static const int                    DEFAULT_B_MAX_FEATURES = 3;
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Compute the keypoints and descriptors for one image (mono)
  bool setMono(int id, const Mat& img);

  // Compute the keypoints, descriptors and 3d points for two images (stereo)
  bool setStereo(int id, const Mat& img_l, const Mat& img_r);

  // Save the camera model
  void setCameraModel(image_geometry::StereoCameraModel stereo_camera_model);

  // Get/set the id
  int getId();
  void setId(int id);

  // Get/set the keypoints of the image (left for stereo)
  vector<KeyPoint> getKp();
  void setKp(vector<KeyPoint> kp);

  // Get/set the descriptors of the image (left for stereo)
  Mat getDesc();
  void setDesc(Mat desc);

  // Get/set the 3D points
  vector<Point3f> get3D();
  void set3D(vector<Point3f> p3d);

  // Bucket features
  vector<DMatch> bucketFeatures(vector<DMatch> matches,
                                vector<KeyPoint> kp);

private:

  Params params_;                       //!> Stores parameters
  image_geometry::StereoCameraModel
    stereo_camera_model_;               //!> Object to save the stereo camera model.
  int id_;                              //!> The current id for this image.
  vector<KeyPoint> kp_;                 //!> Unfiltered keypoints of the images.
  Mat desc_;                            //!> Unfiltered descriptors of the images.
  vector<Point3f> points_3d_;           //!> 3D points of the stereo correspondences.
};

} // namespace

#endif // IMAGE_H