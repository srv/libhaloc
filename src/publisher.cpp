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

#include <ros/ros.h>

#include "libhaloc/publisher.h"

haloc::State::State() {}

haloc::Publisher::Publisher() {
  ros::NodeHandle nhp("~");
  pub_bucketed_img_ = nhp.advertise<sensor_msgs::Image>("bucketed_image", 2, true);
}

void haloc::Publisher::PublishBucketedImage(const State& state,
    const cv::Mat& img) {
  cv::Mat bucketed_img = BuildBucketedImage(state, img);
  cv_bridge::CvImage ros_image;
  ros_image.image = bucketed_img.clone();
  ros_image.header.stamp = ros::Time::now();
  ros_image.encoding = "bgr8";
  pub_bucketed_img_.publish(ros_image.toImageMsg());
}

cv::Mat haloc::Publisher::BuildBucketedImage(const State& state,
    const cv::Mat& img) {
  cv::Mat out_img = img;

  // Draw the bucket lines
  int h_point = static_cast<int>(state.bucket_width);
  while (h_point < img.cols) {
    cv::line(out_img, cv::Point(h_point, 0), cv::Point(h_point, img.rows), cv::Scalar(47, 47, 47), 2, 8);
    h_point += static_cast<int>(state.bucket_width);
  }
  int v_point = static_cast<int>(state.bucket_height);
  while (v_point < img.rows) {
    cv::line(out_img, cv::Point(0, v_point), cv::Point(img.cols, v_point), cv::Scalar(47, 47, 47), 2, 8);
    v_point += static_cast<int>(state.bucket_height);
  }

  // Draw bucket and unbucket keypoints
  cv::drawKeypoints(out_img, state.bucketed_kp, out_img, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(out_img, state.unbucketed_kp, out_img, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  return out_img;
}
