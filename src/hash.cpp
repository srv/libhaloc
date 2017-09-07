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

#include "libhaloc/hash.h"

#include <opencv2/core/eigen.hpp>

haloc::Hash::Params::Params() :
  bucket_rows(DEFAULT_BUCKET_ROWS), bucket_cols(DEFAULT_BUCKET_COLS),
  max_desc(DEFAULT_MAX_DESC), num_proj(DEFAULT_NUM_PROJ)
{}

haloc::Hash::Hash() : initialized_(false) {}

std::vector<float> haloc::Hash::GetHash(
    const std::vector<cv::KeyPoint>& kp, const cv::Mat& desc,
    const cv::Size& img_size) {
  // Initialize first time
  if (!IsInitialized()) Init(img_size, kp.size(), desc.cols);
  state_.Clear();

  // Initialize output
  std::vector<float> hash;

  // The maximum number of features per bucket
  int max_features_x_bucket = static_cast<int>(
    floor(params_.max_desc/(params_.bucket_cols*params_.bucket_rows)));

  // Bucket descriptors
  std::vector<cv::Mat> bucket_desc = BucketDescriptors(kp, desc);

  // Get a hash for every bucket
  const int min_feat = static_cast<int>(0.7 * max_features_x_bucket);
  for (uint i=0; i < bucket_desc.size(); ++i) {
    std::vector<float> bucketed_hash;
    if (bucket_desc[i].rows >= min_feat) {
      bucketed_hash = ProjectDescriptors(bucket_desc[i]);
    } else {
      for (uint i=0; i < desc.cols*params_.num_proj; ++i)
        bucketed_hash.push_back(0.0);
    }
    hash.insert(hash.end(), bucketed_hash.begin(), bucketed_hash.end());
  }
  return hash;
}

int haloc::Hash::CalcDist(const std::vector<float>& hash_a,
    const std::vector<float>& hash_b, float eps) {
  // Init
  const int num_buckets = params_.bucket_cols*params_.bucket_rows;
  // const float eps = 0.6;
  int num_buckets_overlap = 0;

  // Compute the distance
  for (uint i=0; i < comb_.size(); ++i) {
    int comb_overlap = 0;
    for (uint j=0; j < num_buckets; ++j) {
      int idx_a = comb_[i][j].first  * params_.num_proj * desc_length_;
      int idx_b = comb_[i][j].second * params_.num_proj * desc_length_;

      // Check if buckets are empty
      std::vector<float>::const_iterator a_first = hash_a.begin() + idx_a;
      std::vector<float>::const_iterator a_last = hash_a.begin() + idx_a +
        desc_length_*params_.num_proj;
      float sum_a = std::accumulate(a_first, a_last, 0.0);
      if (sum_a == 0.0) continue;

      std::vector<float>::const_iterator b_first = hash_b.begin() + idx_b;
      std::vector<float>::const_iterator b_last = hash_b.begin() + idx_b +
        desc_length_*params_.num_proj;
      float sum_b = std::accumulate(b_first, b_last, 0.0);
      if (sum_b == 0.0) continue;

      float proj_sum = 0.0;
      for (uint k=0; k < desc_length_*params_.num_proj; ++k) {
        proj_sum += fabs(hash_a[idx_a+k] - hash_b[idx_b+k]);
      }
      if (proj_sum <= eps) comb_overlap++;
    }
    if (comb_overlap > num_buckets_overlap) {
      num_buckets_overlap = comb_overlap;
    }
  }
  return num_buckets_overlap;
}

void haloc::Hash::PublishState(const cv::Mat& img) {
  // The bucketed image
  pub_.PublishBucketedImage(state_, img, params_.bucket_rows,
    params_.bucket_cols);

  // The bucketed info
  int max_features_x_bucket = static_cast<int>(
    floor(params_.max_desc/(params_.bucket_cols*params_.bucket_rows)));
  pub_.PublishBucketedInfo(state_, max_features_x_bucket);
}

void haloc::Hash::Init(const cv::Size& img_size, const int& num_feat,
    const int& desc_length) {
  InitProjections(params_.max_desc);
  InitCombinations();
  img_size_ = img_size;
  desc_length_ = desc_length;

  // Sanity check
  if (params_.max_desc < num_feat * 0.7) {
    ROS_WARN_STREAM("[Haloc:] WARNING -> Please setup the maximum number " <<
      "descriptors correctly. The current image has " << num_feat << " and " <<
      "the max_desc param is " << params_.max_desc << ". The parameter for " <<
      "the maximum number of descriptors must be smaller than the number" <<
      "of real features in the images.");
  }

  initialized_ = true;
}

void haloc::Hash::InitCombinations() {
  comb_.clear();
  int num_buckets = params_.bucket_cols*params_.bucket_rows;
  int second_idx_shift = 0;
  for (uint i=0; i < num_buckets; ++i) {
    std::vector< std::pair<int, int> > combinations_row;
    for (uint j=0; j < num_buckets; ++j) {
      int second_idx = j + second_idx_shift;
      if (second_idx > num_buckets-1) {
        second_idx = second_idx - num_buckets;
      }
      combinations_row.push_back(std::make_pair(j, second_idx));
    }
    comb_.push_back(combinations_row);
    second_idx_shift++;
  }
}

void haloc::Hash::InitProjections(const int& desc_size) {
  // Initializations
  int seed = time(NULL);
  r_.clear();

  // The maximum number of features per bucket
  int max_features_x_bucket = static_cast<int>(
    floor(params_.max_desc/(params_.bucket_cols*params_.bucket_rows)));

  // The size of the descriptors may vary...
  // But, we limit the number of descriptors per bucket.
  int v_size = max_features_x_bucket;

  // We will generate N-orthogonal vectors creating a linear system of type Ax=b
  // Generate a first random vector
  std::vector<float> r = ComputeRandomVector(v_size, seed);
  r_.push_back(UnitVector(r));

  // Generate the set of orthogonal vectors
  for (uint i=1; i < params_.num_proj; i++) {
    // Generate a random vector of the correct size
    std::vector<float> new_v = ComputeRandomVector(v_size - i, seed + i);

    // Get the right terms (b)
    Eigen::VectorXf b(r_.size());
    for (uint n=0; n < r_.size(); n++) {
      std::vector<float> cur_v = r_[n];
      float sum = 0.0;
      for (uint m=0; m < new_v.size(); m++)
        sum += new_v[m]*cur_v[m];
      b(n) = -sum;
    }

    // Get the matrix of equations (A)
    Eigen::MatrixXf A(i, i);
    for (uint n=0; n < r_.size(); n++) {
      uint k = 0;
      for (uint m=r_[n].size()-i; m < r_[n].size(); m++) {
        A(n, k) = r_[n][m];
        k++;
      }
    }

    // Apply the solver
    Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);

    // Add the solutions to the new vector
    for (uint n=0; n < r_.size(); n++)
      new_v.push_back(x(n));
    new_v = UnitVector(new_v);

    // Push the new vector
    r_.push_back(new_v);
  }
}

std::vector<float> haloc::Hash::ComputeRandomVector(const int& size, int seed) {
  std::vector<float> h;
  for (int i=0; i < size; i++)
    h.push_back((float)rand()/RAND_MAX);
  return h;
}

std::vector<float> haloc::Hash::UnitVector(const std::vector<float>& x) {
  // Compute the norm
  float sum = 0.0;
  for (uint i=0; i < x.size(); i++)
    sum += pow(x[i], 2.0);
  float x_norm = sqrt(sum);

  // x^ = x/|x|
  std::vector<float> out;
  for (uint i=0; i < x.size(); i++)
    out.push_back(x[i] / x_norm);

  return out;
}

std::vector<cv::Mat> haloc::Hash::BucketDescriptors(
    const std::vector<cv::KeyPoint>& kp, const cv::Mat& desc) {
  // Find max values
  float u_max = 0;
  float v_max = 0;
  std::vector<cv::KeyPoint> kp_in = kp;
  for (std::vector<cv::KeyPoint>::iterator it = kp_in.begin();
      it != kp_in.end(); it++) {
    if (it->pt.x > u_max) u_max = it->pt.x;
    if (it->pt.y > v_max) v_max = it->pt.y;
  }

  // Compute width and height of the buckets
  float bucket_width  = img_size_.width / params_.bucket_cols;
  float bucket_height = img_size_.height / params_.bucket_rows;

  // Allocate number of buckets needed
  std::vector<cv::Mat> desc_buckets(params_.bucket_cols*params_.bucket_rows);
  std::vector< std::vector<cv::KeyPoint> > kp_buckets(
    params_.bucket_cols*params_.bucket_rows);

  // Assign descriptors to their buckets
  for (uint i=0; i < kp_in.size(); ++i) {
    int u = static_cast<int>(floor(kp_in[i].pt.x/bucket_width));
    int v = static_cast<int>(floor(kp_in[i].pt.y/bucket_height));

    desc_buckets[v*params_.bucket_cols+u].push_back(desc.row(i));
    kp_buckets[v*params_.bucket_cols+u].push_back(kp_in[i]);
  }

  // The maximum number of features per bucket
  int max_features_x_bucket = static_cast<int>(
    floor(params_.max_desc/(params_.bucket_cols*params_.bucket_rows)));

  // Select the best keypoints for each bucket
  std::vector<cv::Mat> out_desc(params_.bucket_cols*params_.bucket_rows);
  for (int i=0; i < params_.bucket_cols*params_.bucket_rows; ++i) {
    // Sort keypoints by response
    std::vector<cv::KeyPoint> cur_kps = kp_buckets[i];
    std::vector<int> index(cur_kps.size(), 0);
    for (uint j=0; j<index.size(); j++) {
      index[j] = j;
    }
    std::sort(index.begin(), index.end(), [&](const int& a, const int& b) {
      return (cur_kps[a].response > cur_kps[b].response);
    });

    // Add up to max_features_x_bucket features from this bucket
    int k = 0;
    int num_kp = 0;
    for (uint j=0; j < cur_kps.size(); ++j) {
      if (k < max_features_x_bucket) {
        out_desc[i].push_back(desc_buckets[i].row(index[j]));
        state_.bucketed_kp.push_back(kp_buckets[i][index[j]]);
        num_kp++;
      } else {
        state_.unbucketed_kp.push_back(kp_buckets[i][index[j]]);
      }
      k++;
    }
    state_.num_kp_per_bucket.push_back(num_kp);
  }

  return out_desc;
}

std::vector<float> haloc::Hash::ProjectDescriptors(const cv::Mat& desc) {
  // Initialize first time
  if (!IsInitialized()) Init(cv::Size(0, 0), desc.rows, desc.cols);

  // Initialize output
  std::vector<float> hash;

  // Sanity checks
  if (desc.rows == 0) {
    ROS_ERROR("[Haloc:] ERROR -> Descriptor matrix is empty.");
    return hash;
  }

  if (desc.rows > r_[0].size()) {
    ROS_ERROR_STREAM("[Haloc:] ERROR -> The number of descriptors is " <<
      "larger than the size of the projection vector. This should not happen.");
    return hash;
  }

  // Project the descriptors
  for (uint i=0; i < r_.size(); i++) {
    for (int n=0; n < desc.cols; n++) {
      float desc_sum = 0.0;
      for (uint m=0; m < desc.rows; m++) {
        float projected = r_[i][m]*desc.at<float>(m, n);
        float projected_normalized = (projected + 1.0) / 2.0;
        desc_sum += projected_normalized;
      }
      hash.push_back(desc_sum / static_cast<float>(desc.rows));
    }
  }
  return hash;
}
