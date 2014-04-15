#ifndef LC_H
#define LC_H

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <image_geometry/stereo_camera_model.h>
#include "hash.h"
#include "image.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace haloc
{

class LoopClosure
{

public:

  // Class contructor
  LoopClosure();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    string work_dir;                    //!> Directory where the library will save the image informations (must be writtible!).
    string desc_type;                   //!> Type of the descriptors (can be SIFT, SURF).
    int num_proj;                       //!> Number of projections for image hashing
    double desc_thresh;                 //!> Descriptor threshold (tipically between 0.7-0.9).
    int epipolar_thresh;                //!> Epipolar threshold.
    int min_neighbour;                  //!> Minimum number of neighbours that will be skiped for the loop closure (tipically between 5-20, but depends on the frame rate).
    int n_candidates;                   //!> Get the n first candidates of the hash matching (tipically between 1-5).
    int min_matches;                    //!> Minimun number of descriptor matches to consider a matching as possible loop closure (>8).
    int min_inliers;                    //!> Minimum number of inliers to consider a matching as possible loop closure (>8).
    double max_reproj_err;              //!> Maximum reprojection error (stereo only).
    bool validate;                      //!> True if you want to validate the loop closure (spends more time). Default False.

    // Default values
    static const int                    DEFAULT_NUM_PROJ = 2;
    static const double                 DEFAULT_DESC_THRESH = 0.8;
    static const int                    DEFAULT_EPIPOLAR_THRESH = 1;
    static const int                    DEFAULT_MIN_NEIGHBOUR = 10;
    static const int                    DEFAULT_N_CANDIDATES = 2;
    static const int                    DEFAULT_MIN_MATCHES = 20;
    static const int                    DEFAULT_MIN_INLIERS = 12;
    static const double                 DEFAULT_MAX_REPROJ_ERR = 3.0;
    static const bool                   DEFAULT_VALIDATE = false;
  };

  // Set the parameter struct.
  void setParams(const Params& params);

  // Return current parameters.
  inline Params params() const { return params_; }

  // Initialize class.
  void init();

  // Save the camera model.
  void setCameraModel(image_geometry::StereoCameraModel stereo_camera_model, Mat camera_matrix);

  // Compute kp, desc and hash for one image (mono verion).
  void setNode(Mat img);

  // Compute kp, desc and hash for two images (stereo verion).
  void setNode(Mat img_l, Mat img_r);

  // Try to find a loop closure for the last saved node.
  bool getLoopClosure(int& lc_img_idx);
  bool getLoopClosure(int& lc_img_idx, tf::Transform& trans);

private:

  // Compute the loop closure
  bool compute(Image ref_image, string cur_filename, int &matches, int &inliers, tf::Transform& trans);

  // Properties
  Params params_;                       //!> Stores parameters
  Image img_;                           //!> Image object
  Hash hash_;                           //!> Hash object
  int img_idx_;                         //!> Incremental index for the stored images
  Mat camera_matrix_;                   //!> Used to save the camera matrix
  vector< pair<int, vector<float> > > hash_table_;  //!> Hash table

};

} // namespace

#endif // LC_H