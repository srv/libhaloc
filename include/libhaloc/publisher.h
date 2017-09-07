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
#include <std_msgs/String.h>

#include <vector>

#include "libhaloc/state.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace haloc {

class Publisher {
 public:
  /**
   * @brief      Empty class constructor.
   */
  Publisher();

  /**
   * @brief      Publishes the bucketed image.
   *
   * @param[in]  state        The state obtained after a hash computation.
   * @param[in]  img          The original image.
   * @param[in]  bucket_rows  The bucket rows
   * @param[in]  bucket_cols  The bucket cols
   */
  void PublishBucketedImage(const State& state, const cv::Mat& img,
    const int& bucket_rows, const int& bucket_cols);

  /**
   * @brief      Publishes the bucketed info
   *
   * @param[in]  state        The state obtained after a hash computation.
   * @param[in]  max_feat     The maximum number of features per bucket
   */
  void PublishBucketedInfo(const State& state, const int& max_feat);

 protected:
  /**
   * @brief      Returns a debug image with the bucketed keypoints.
   *
   * @param[in]  state        The state obtained after a hash computation.
   * @param[in]  img          The original image.
   * @param[in]  bucket_rows  The bucket rows
   * @param[in]  bucket_cols  The bucket cols
   *
   * @return     The bucketed image.
   */
  cv::Mat BuildBucketedImage(const State& state, const cv::Mat& img,
    const int& bucket_rows, const int& bucket_cols);

 private:

  // The ROS publishers
  ros::Publisher pub_bucketed_img_;
  ros::Publisher pub_bucketed_info_;

};

}  // namespace haloc

#endif  // LIBHALOC_INCLUDE_LIBHALOC_PUBLISHER_H_
