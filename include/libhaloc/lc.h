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

  // Class constructor
  LoopClosure();

  struct Params
  {
    //Default constructor sets all values to defaults.
    Params();

    // Class parameters
    int num_proj;                       //!> Number of projections for image hashing
    string work_dir;                    //!> Directory where the library will save the image informations (must be writable!).
    string desc_type;                   //!> Type of the descriptors (can be SIFT, SURF).
    string desc_matching_type;          //!> Can be "CROSSCHECK" or "RATIO"
    double desc_thresh_ratio;           //!> Descriptor threshold for crosscheck matching (typically between 0.7-0.9) or ratio for ratio matching (typically between 0.6-0.8).
    int epipolar_thresh;                //!> Epipolar threshold.
    int min_neighbour;                  //!> Minimum number of neighbours that will be skipped for the loop closure (typically between 5-20, but depends on the frame rate).
    int n_candidates;                   //!> Get the n first candidates of the hash matching (typically between 1-5).
    int min_matches;                    //!> Minimum number of descriptor matches to consider a matching as possible loop closure (>8).
    int min_inliers;                    //!> Minimum number of inliers to consider a matching as possible loop closure (>8).
    double max_reproj_err;              //!> Maximum reprojection error (stereo only).
    bool validate;                      //!> True if you want to validate the loop closure (spends more time). Default False.
    bool verbose;                       //!> Set to true to show logs in the screen.

    // Default values
    static const int                    DEFAULT_NUM_PROJ = 2;
    static const double                 DEFAULT_DESC_THRESH_RATIO = 0.8;
    static const int                    DEFAULT_EPIPOLAR_THRESH = 1;
    static const int                    DEFAULT_MIN_NEIGHBOUR = 10;
    static const int                    DEFAULT_N_CANDIDATES = 2;
    static const int                    DEFAULT_MIN_MATCHES = 20;
    static const int                    DEFAULT_MIN_INLIERS = 12;
    static const double                 DEFAULT_MAX_REPROJ_ERR = 2.0;
    static const bool                   DEFAULT_VALIDATE = false;
    static const bool                   DEFAULT_VERBOSE = false;
  };

  // Set the parameter struct.
  void setParams(const Params& params);

  // Return current parameters.
  inline Params params() const { return params_; }

  // Initialize class.
  void init();

  // Finalize class.
  void finalize();

  // Save the camera model.
  void setCameraModel(image_geometry::StereoCameraModel stereo_camera_model,
                      Mat camera_matrix);

  // Compute kp, desc and hash for one image (mono version).
  void setNode(Mat img,
               string name);

  // Compute kp, desc and hash for two images (stereo version).
  void setNode(Mat img_l,
               Mat img_r,
               string name);

  // Try to find a loop closure for the last saved node.
  bool getLoopClosure(int& lc_img_idx,
                      string& lc_name);
  bool getLoopClosure(int& lc_img_idx,
                      string& lc_name,
                      tf::Transform& trans);

  // Try to find a loop closure given 2 image identifiers
  bool getLoopClosureById(string image_id_ref,
                          string image_id_cur,
                          tf::Transform& trans);

private:

  // Compute the loop closure
  bool compute(Image ref_image,
               string cur_filename,
               string &lc_name,
               int &matches,
               int &inliers,
               tf::Transform& trans);

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