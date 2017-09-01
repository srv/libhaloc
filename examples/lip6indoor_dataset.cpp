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

#include <boost/filesystem.hpp>

#include <opencv2/opencv.hpp>

#include "libhaloc/hash.h"

namespace fs = boost::filesystem;

/**
 * @brief      Shows the program usage.
 *
 * @param[in]  name  The program name
 */
static void ShowUsage(const std::string& name) {
  std::cerr << "Usage: " << name << " image_directory" << std::endl;
}

/**
 * @brief      Main entry point
 *
 * @param[in]  argc  The argc
 * @param      argv  The argv
 *
 * @return     0
 */
int main(int argc, char** argv) {
  // Parse arguments
  std::string img_dir = "";
  if (argc != 1 && argc != 2) {
    ShowUsage(argv[0]);
    return 0;
  }
  if (argc == 2) {
    img_dir = argv[1];
  } else {
    ShowUsage(argv[0]);
    return 0;
  }

  ros::init(argc, argv, "lip6indoor_dataset");

  // Init feature extractor
  cv::Ptr<cv::Feature2D> feat(new cv::Feature2D());
  feat = cv::KAZE::create();

  // Hash object
  haloc::Hash haloc;

  // Set params
  haloc::Hash::Params params;
  params.max_desc = 45;
  haloc.SetParams(params);

  // Sort directory of images
  typedef std::vector<fs::path> vec;
  vec v;
  copy(fs::directory_iterator(img_dir), fs::directory_iterator(),
    back_inserter(v));
  sort(v.begin(), v.end());
  vec::const_iterator it(v.begin());

  ROS_INFO_STREAM("Processing directory: " << img_dir << " with " <<
    v.size() << " images.");

  // Loop over the images
  while (it != v.end()) {
    // Check if the directory entry is an directory.
    if (fs::is_directory(*it)) {
      it++;
      continue;
    }

    // Open the image
    std::string filename = it->filename().string();
    std::string path = img_dir + "/" + filename;
    cv::Mat img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    ROS_INFO_STREAM("Reading image " << filename);

    // Extract keypoints and descriptors
    cv::Mat desc;
    std::vector<cv::KeyPoint> kp;
    feat->detectAndCompute(img, cv::noArray(), kp, desc);
    ROS_INFO_STREAM("Number of features: " << kp.size());

    // Compute the hash
    haloc.GetBucketedHash(kp, desc, img.size());
    haloc.PublishState(img);
    ros::WallDuration(5).sleep();

    it++;
  }
  return 0;
}
