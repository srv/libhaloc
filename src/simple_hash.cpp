#include <ros/ros.h>
#include <ros/package.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>

#include "libhaloc/simple_hash.h"

namespace haloc
{

  SimpleHash::Params::Params() :
    num_proj(DEFAULT_NUM_PROJ)
  {}

  SimpleHash::SimpleHash()
  {
    // Initializations
    h_size_ = -1;
    initialized_ = false;
  }

  bool SimpleHash::isInitialized()
  {
    return initialized_;
  }

  void SimpleHash::setParams(const Params& params)
  {
    params_ = params;
  }

  void SimpleHash::init(cv::Mat desc)
  {
    // Create the random projections vectors
    initProjections(desc.rows);

    // Set the size of the descriptors
    h_size_ = params_.num_proj * desc.cols;

    // Set initialized to true
    initialized_ = true;
  }

  vector<float> SimpleHash::getHash(cv::Mat desc)
  {
    // Initialize the histogram with 0's
    vector<float> hash(h_size_, 0.0);

    // Sanity check
    if (desc.rows == 0) return hash;

    // Project the descriptors
    uint k = 0;
    for (uint i=0; i<r_.size(); i++)
    {
      for (int n=0; n<desc.cols; n++)
      {
        float desc_sum = 0.0;
        for (int m=0; m<desc.rows; m++)
          desc_sum += r_[i][m]*desc.at<float>(m, n);

        hash[k] = desc_sum/(float)desc.rows;
        k++;
      }
    }

    return hash;
  }

  void SimpleHash::initProjections(int desc_size)
  {
    // Initializations
    int seed = time(NULL);
    r_.clear();

    // The size of the descriptors may vary... We multiply the current descriptor size
    // for a scalar to handle the larger cases.
    int v_size = 6*desc_size;

    // We will generate N-orthogonal vectors creating a linear system of type Ax=b.

    // Generate a first random vector
    vector<float> r = compute_random_vector(seed, v_size);

    // Make it CTS
    float acc = 0.0;
    for (uint i=0; i<r.size(); i++)
      acc += r[i];
    float mean = acc / r.size();
    for (uint i=0; i<r.size(); i++)
      r[i] = mean;

    // r_.push_back(unit_vector(r));
    r_.push_back(r);

    // Generate the set of orthogonal vectors
    for (int i=1; i<params_.num_proj; i++)
    {
      // Generate a random vector of the correct size
      vector<float> new_v = compute_random_vector(seed + i, v_size - i);

      // // Make it CTS
      // acc = 0.0;
      // for (uint i=0; i<new_v.size(); i++)
      //   acc += new_v[i];
      // mean = acc / new_v.size();
      // for (uint i=0; i<new_v.size(); i++)
      //   new_v[i] = mean;

      // Get the right terms (b)
      VectorXf b(r_.size());
      for (uint n=0; n<r_.size(); n++)
      {
        vector<float> cur_v = r_[n];
        float sum = 0.0;
        for (uint m=0; m<new_v.size(); m++)
        {
          sum += new_v[m]*cur_v[m];
        }
        b(n) = -sum;
      }

      // Get the matrix of equations (A)
      MatrixXf A(i, i);
      for (uint n=0; n<r_.size(); n++)
      {
        uint k=0;
        for (uint m=r_[n].size()-i; m<r_[n].size(); m++)
        {
          A(n,k) = r_[n][m];
          k++;
        }
      }

      // Apply the solver
      VectorXf x = A.colPivHouseholderQr().solve(b);

      // Add the solutions to the new vector
      for (uint n=0; n<r_.size(); n++)
        new_v.push_back(x(n));
      new_v = unit_vector(new_v);

      // Push the new vector
      r_.push_back(new_v);
    }

    ostringstream proj_csv;
    for (uint i=0; i<r_[0].size(); i++)
    {
      for (uint j=0; j<r_.size(); j++)
        proj_csv << r_[j][i] << ",";
      proj_csv.seekp(proj_csv.str().length()-1);
      proj_csv << endl;
    }

    // Save projections to file
    string out_file = ros::package::getPath("libhaloc") + "/projections.txt";
    fstream f_out(out_file.c_str(), ios::out | ios::trunc);
    f_out << proj_csv.str();
    f_out.close();
  }

  vector<float> SimpleHash::compute_random_vector(uint seed, int size)
  {
    srand(seed);
    vector<float> h;
    for (int i=0; i<size; i++)
      h.push_back( (float)rand()/RAND_MAX );
    return h;
  }

  vector<float> SimpleHash::unit_vector(vector<float> x)
  {
    // Compute the norm
    float sum = 0.0;
    for (uint i=0; i<x.size(); i++)
      sum += pow(x[i], 2.0);
    float x_norm = sqrt(sum);

    // x^ = x/|x|
    for (uint i=0; i<x.size(); i++)
      x[i] = x[i] / x_norm;

    return x;
  }

  float SimpleHash::match(vector<float> hash_1, vector<float> hash_2)
  {
    // Compute the distance
    float sum = 0.0;
    for (uint i=0; i<hash_1.size(); i++)
      sum += fabs(hash_1[i] - hash_2[i]);

    return sum;
  }

  vector<KeyPoint> SimpleHash::bucketFeatures(vector<KeyPoint> kp)
  {
    int b_width = 30;
    int b_height = 30;
    int num_feat = 7;

    // Find max values
    float x_max = 0;
    float y_max = 0;
    for (uint i=0; i<kp.size(); i++)
    {
      if (kp[i].pt.x > x_max) x_max = kp[i].pt.x;
      if (kp[i].pt.y > y_max) y_max = kp[i].pt.y;
    }

    // Allocate number of buckets needed
    int bucket_cols = (int)floor(x_max/b_width) + 1;
    int bucket_rows = (int)floor(y_max/b_height) + 1;
    vector<KeyPoint> *buckets = new vector<KeyPoint>[bucket_cols*bucket_rows];

    // Assign kp to their buckets
    for (uint i=0; i<kp.size(); i++)
    {
      int u = (int)floor(kp[i].pt.x/b_width);
      int v = (int)floor(kp[i].pt.y/b_height);
      buckets[v*bucket_cols+u].push_back(kp[i]);
    }

    // Refill kp from buckets
    vector<KeyPoint> output;
    for (int i=0; i<bucket_cols*bucket_rows; i++)
    {
      // Sort descriptors matched by distance
      sort(buckets[i].begin(), buckets[i].end(), haloc::Utils::sortByResponse);

      // Add up to max_features features from this bucket to output
      int k=0;
      for (vector<KeyPoint>::iterator it=buckets[i].begin(); it!=buckets[i].end(); it++)
      {
        output.push_back(*it);
        k++;
        if (k >= num_feat)
          break;
      }
    }
    return output;
  }

} //namespace slam
