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

#ifndef LIBHALOC_INCLUDE_LIBHALOC_PUBLISHER_H_
#define LIBHALOC_INCLUDE_LIBHALOC_PUBLISHER_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

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
    bucket_width = 0;
    bucket_height = 0;
  }

  // State variables
  std::vector<cv::KeyPoint> bucketed_kp;    //!> Stores the bucketed keypoints
  std::vector<cv::KeyPoint> unbucketed_kp;  //!> Stores the discarded keypoints when bucketing
  float bucket_width;                       //!> Stores the bucket width
  float bucket_height;                      //!> Stores the bucket height
};

class Publisher {
 public:
  /**
   * @brief      Empty class constructor.
   */
  Publisher();

  /**
   * @brief      Publishes the bucketed image.
   *
   * @param[in]  state  The state obtained after a hash computation.
   * @param[in]  img    The original image.
   */
  void PublishBucketedImage(const State& state, const cv::Mat& img);

 protected:
  /**
   * @brief      Returns a debug image with the bucketed keypoints.
   *
   * @param[in]  state  The state obtained after a hash computation.
   * @param[in]  img    The original image.
   *
   * @return     The bucketed image.
   */
  cv::Mat BuildBucketedImage(const State& state, const cv::Mat& img);

 private:

  // The ROS publishers
  ros::Publisher pub_bucketed_img_;

};

}  // namespace haloc

#endif  // LIBHALOC_INCLUDE_LIBHALOC_PUBLISHER_H_
