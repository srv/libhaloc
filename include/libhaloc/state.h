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

#ifndef LIBHALOC_INCLUDE_LIBHALOC_STATE_H_
#define LIBHALOC_INCLUDE_LIBHALOC_STATE_H_

#include <Eigen/Eigen>
#include <Eigen/Dense>

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace haloc {

/**
 * @brief      Struct to save operational variables after every hash
 *             computation.
 */
struct State {
  /**
   * @brief      Default constructor.
   */
  State();

  void Clear() {
    bucketed_kp.clear();
    unbucketed_kp.clear();
    num_kp_per_bucket.clear();
  }

  // State variables
  std::vector<cv::KeyPoint> bucketed_kp;    //!> Stores the bucketed keypoints
  std::vector<cv::KeyPoint> unbucketed_kp;  //!> Stores the discarded keypoints when bucketing
  std::vector<int> num_kp_per_bucket;       //!> The number of keypoints per bucket
};

}  // namespace haloc

#endif  // LIBHALOC_INCLUDE_LIBHALOC_STATE_H_
