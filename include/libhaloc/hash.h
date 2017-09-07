//  Copyright (c) 2017 Universitat de les Illes Balears
//  This file is part of LIBHALOC.
//
//  LIBHALOC is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  LIBHALOC is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with LIBHALOC. If not, see <http://www.gnu.org/licenses/>.

#ifndef LIBHALOC_INCLUDE_LIBHALOC_HASH_H_
#define LIBHALOC_INCLUDE_LIBHALOC_HASH_H_

#include <Eigen/Eigen>
#include <Eigen/Dense>

#include <vector>
#include <utility>
#include <numeric>

#include "libhaloc/publisher.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace haloc {

class Hash {
 public:
  /**
   * @brief      Struct for class parameters
   */
  struct Params {
    /**
     * @brief      Default constructor
     */
    Params();

    // Class parameters
    int bucket_rows;             //!> Number of horizontal divisions for the descriptors bucketing
    int bucket_cols;             //!> Number of vertical divisions for the descriptors bucketing
    int max_desc;                //!> Maximum number of descriptors per image
    int num_proj;                //!> Number of projections required

    // Default values
    static const int             DEFAULT_BUCKET_ROWS = 3;
    static const int             DEFAULT_BUCKET_COLS = 4;
    static const int             DEFAULT_MAX_DESC = 100;
    static const int             DEFAULT_NUM_PROJ = 2;
  };

  /**
   * @brief      Empty class constructor.
   */
  Hash();

  /**
   * @brief      Sets the parameters.
   *
   * @param[in]  params  The parameters.
   */
  inline void SetParams(const Params& params) {
    params_ = params; initialized_ = false;}

  /**
   * @brief      Returns the parameters.
   *
   * @return     The parameters.
   */
  inline Params GetParams() const {return params_;}

  /**
   * @brief      Determines if class is initialized.
   *
   * @return     True if initialized, False otherwise.
   */
  inline bool IsInitialized() const {return initialized_;}

  /**
   * @brief      Gets the state.
   *
   * @return     The state.
   */
  inline State GetState() const {return state_;}

  /**
   * @brief      Bucket the features and compute a hash for every bucket.
   *
   * @param[in]  kp        The keypoints vector.
   * @param[in]  desc      The descriptors.
   * @param[in]  img_size  The image size.
   *
   * @return     The bucketed hash.
   */
  std::vector<float> GetHash(const std::vector<cv::KeyPoint>& kp,
    const cv::Mat& desc, const cv::Size& img_size);

  /**
   * @brief      Compute the distance between 2 hashes.
   *
   * @param[in]  hash_1  The hash 1.
   * @param[in]  hash_2  The hash 2.
   *
   * @return     Distance: the number of buckets seeing the same view.
   */
  int CalcDist(const std::vector<float>& hash_1,
    const std::vector<float>& hash_2, float eps);

  /**
   * @brief      Publishes the state and debug variables. Must be called after a
   *             hash computation.
   *
   * @param[in]  img   The original image
   */
  void PublishState(const cv::Mat& img);

 protected:
  /**
   * @brief      Init the class.
   *
   * @param[in]  img_size     The image size.
   * @param[in]  num_feat     The number of features for the input image.
   * @param[in]  desc_length  The descriptor length
   */
  void Init(const cv::Size& img_size, const int& num_feat,
    const int& desc_length);

  /**
   * @brief      Compute the combinations required for the match calculation
   */
  void InitCombinations();

  /**
   * @brief      Initializes the random vectors for projections.
   *
   * @param[in]  size  The size.
   */
  void InitProjections(const int& size);

  /**
   * @brief      Calculates a random vector.
   *
   * @param[in]  size  The size.
   * @param[in]  seed  The seed.
   *
   * @return     The random vector.
   */
  std::vector<float> ComputeRandomVector(const int& size, int seed);

  /**
   * @brief      Makes a vector unitary.
   *
   * @param[in]  x     The input vector.
   *
   * @return     The output unit vector.
   */
  std::vector<float> UnitVector(const std::vector<float>& x);

  /**
   * @brief      Bucket the descriptors.
   *
   * @param[in]  kp    The keypoint vector.
   * @param[in]  desc  The descriptors.
   *
   * @return     The bucketed descriptors.
   */
  std::vector<cv::Mat> BucketDescriptors(const std::vector<cv::KeyPoint>& kp,
    const cv::Mat& desc);

  /**
   * @brief      Compute the hash by projecting the descriptors.
   *
   * @param[in]  desc  The descriptors.
   *
   * @return     The hash.
   */
  std::vector<float> ProjectDescriptors(const cv::Mat& desc);

 private:
  // Properties
  Params params_;                        //!> Stores parameters
  State state_;                          //!> Stores the state after every hash computation
  cv::Size img_size_;                    //!> Image size (only needed for bucketing)
  int desc_length_;                      //!> The length of the descriptors used
  std::vector< std::vector<float> > r_;  //!> Vector of random values
  bool initialized_;                     //!> True when class has been initialized
  std::vector< std::vector< std::pair<int, int> > > comb_;  //!> Combinations for the match
  Publisher pub_;                        //!> The publisher for debugging purposes
};

}  // namespace haloc

#endif  // LIBHALOC_INCLUDE_LIBHALOC_HASH_H_
