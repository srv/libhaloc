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

  // Class contructor
  Image();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    string desc_type;                   //!> Type of the descriptors (can be SIFT, SURF).
    double desc_thresh;                 //!> Descriptor threshold (tipically between 0.7-0.9). Stereo only.
    int epipolar_thresh;                //!> Epipolar threshold (stereo only).

    // Default values
    static const double                 DEFAULT_DESC_THRESH = 0.8;
    static const int                    DEFAULT_EPIPOLAR_THRESH = 1;
  };

  // Set the parameter struct
  void setParams(const Params& params);

  // Return current parameters
  inline Params params() const { return params_; }

  // Compute the keypoints and descriptors for one image (mono)
  void setMono(const Mat& img, string name);

  // Compute the keypoints and descriptors for two images (stereo)
  void setStereo(const Mat& img_l, const Mat& img_r, string name);

  // Save the camera model
  void setCameraModel(image_geometry::StereoCameraModel stereo_camera_model);

  // Get/set the keypoints of the image (left for stereo)
  vector<Point2f> getKp();
  void setKp(vector<Point2f>);

  // Get/set the descriptors of the image (left for stereo)
  Mat getDesc();
  void setDesc(Mat);

  // Get/set the 3D points
  vector<Point3f> get3D();
  void set3D(vector<Point3f>);

  // Get/set the name
  string getName();
  void setName(string);

private:

  Params params_;                       //!> Stores parameters

  image_geometry::StereoCameraModel
    stereo_camera_model_;               //!> Object to save the stereo camera model.

  vector<Point2f> kp_;                  //!> Unfiltered keypoints of the images.
  Mat desc_;                            //!> Unfiltered descriptors of the images.
  vector<Point3f> points_3d_;           //!> 3D points of the stereo correspondences.
  string name_;                         //!> Name identifying the current image
};

} // namespace

#endif // IMAGE_H