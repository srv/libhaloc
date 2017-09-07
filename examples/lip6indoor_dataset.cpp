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

#include <fstream>

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

  // Init ROS
  ros::init(argc, argv, "lip6indoor_dataset");

  // Init feature extractor
  cv::Ptr<cv::Feature2D> feat(new cv::Feature2D());
  feat = cv::KAZE::create();

  // Hash object
  haloc::Hash haloc;

  // Set params
  haloc::Hash::Params params;
  params.max_desc = 100;
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

  // Operational variables
  int img_idx = 0;
  int discard_window = 10;
  std::map<int, std::vector<float> > hash_table;

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
    std::vector<float> hash = haloc.GetHash(kp, desc, img.size());
    hash_table.insert(std::pair<int, std::vector<float> >(img_idx, hash));

    // Log
    haloc.PublishState(img);
    haloc::State state = haloc.GetState();
    ROS_INFO_STREAM("Number of features after bucketing: " <<
      state.bucketed_kp.size());

    img_idx++;
    it++;
    std::cout << std::endl;
  }

  // Find loop closings
  ROS_INFO("Generating the output matrix...");
  for (float eps=0.6; eps <= 1.0; eps=eps+0.02) {
    std::stringstream hist_file, output_file;
    hist_file << "/tmp/histogram" << eps << ".csv";
    output_file << "/tmp/output" << eps << ".csv";
    std::fstream hist(hist_file.str().c_str(), std::ios::out);
    std::fstream output(output_file.str().c_str(), std::ios::out);
    hist << std::fixed << std::setprecision(10);
    output << std::fixed << std::setprecision(10);

    ROS_INFO_STREAM("Processing eps: " << eps);

    int great_3 = 0;
    int great_4 = 0;
    int great_5 = 0;
    int great_6 = 0;

    for (uint i=0; i < hash_table.size(); ++i) {
      for (uint j=0; j < hash_table.size(); ++j) {
        int dist_original = 0;
        int dist = 0;
        int neighbourhood = abs(i-j);
        if (neighbourhood > 20 && j < i) {
          dist_original = haloc.CalcDist(hash_table[i], hash_table[j], eps);
          if (dist_original > 3) great_3++;
          if (dist_original > 4) great_4++;
          if (dist_original > 5) great_5++;
          if (dist_original > 6) great_6++;

          if (dist_original < 4) {
            dist = 0;
          } else {
            dist = 1;
          }
        }

        // Log
        hist << dist_original << ", ";

        // Log
        if (j != hash_table.size()-1) {
          output << dist << ", ";
        } else {
          output << dist << std::endl;
        }
      }
    }
    hist.close();
    output.close();

    ROS_INFO_STREAM("  >3: " << great_3);
    ROS_INFO_STREAM("  >4: " << great_4);
    ROS_INFO_STREAM("  >5: " << great_5);
    ROS_INFO_STREAM("  >6: " << great_6);

  }

  ROS_INFO_STREAM("Finished!");

  ros::shutdown();
  return 0;
}
