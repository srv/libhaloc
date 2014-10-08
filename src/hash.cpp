#include "libhaloc/hash.h"
#include <opencv2/core/eigen.hpp>


/** \brief Hash class constructor
  */
haloc::Hash::Hash() : bucket_max_size_(200), desc_length_(-1)
{
  // Initializations
  initialized_ = false;
}

/** \brief Return true if the class has been initialized
  * @return
  */
bool haloc::Hash::isInitialized()
{
  return initialized_;
}

// Class initialization
void haloc::Hash::init()
{
  // Create the random projections vectors
  initProjections();

  // Set initialized to true
  initialized_ = true;
}

/** \brief Compute the hash
  * @return hash vector
  * \param cvMat containing the descriptors of the image
  */
haloc::Hash::HashBucket haloc::Hash::getHash(vector<Mat> desc)
{
  // Init the output
  haloc::Hash::HashBucket hash_bucket;

  // Get the length of the descriptors
  if (desc_length_ == -1)
  {
    for (uint i=0; i<desc.size(); i++)
    {
      if (desc[i].cols > 0)
      {
        desc_length_ = desc[i].cols;
        break;
      }
    }
  }

  // Sanity check
  if (desc_length_ == -1 || desc.size() == 0) return hash_bucket;

  // Initialize the hash properties
  Mat hash_value(desc.size(), desc_length_, CV_32F);
  vector<int> num_features;

  // Get a hash for every bucket
  for (uint i=0; i<desc.size(); i++)
  {
    // Take the projection vector and the descriptors for this bucket
    vector<float> br = r_[i];
    Mat b_desc = desc[i];

    // Check if there are descriptors into this bucket
    if (b_desc.rows == 0)
    {
      num_features.push_back(0);
      Mat zeros = Mat::zeros(1, desc_length_, CV_32F);
      zeros.copyTo(hash_value.row(i));
      continue;
    }

    // Sanity check
    int max_features = b_desc.rows;
    if (max_features > bucket_max_size_) max_features = bucket_max_size_;

    // Project the descriptors
    vector<float> b_hash_value;
    for (int n=0; n<b_desc.cols; n++)
    {
      float desc_sum = 0.0;
      for (uint m=0; m<max_features; m++)
      {
        desc_sum += br[m]*b_desc.at<float>(m, n);
      }
      hash_value.at<float>(i, n) = desc_sum/(float)max_features;
    }
    num_features.push_back(max_features);
  }

  // Build the bucketed hash structure and exit
  hash_bucket.num_features = num_features;
  hash_bucket.hash_value = hash_value;
  return hash_bucket;
}

/** \brief Compute the random vector/s needed to generate the hash
  * @return
  * \param size of the initial descriptor matrix.
  * \param true to generate 'n' random orthogonal projections, false to generate 'n' random non-orthogonal projections.
  */
void haloc::Hash::initProjections()
{
  // Initializations
  int seed = time(NULL);
  r_.clear();

  // Get a random unit vector for every bucket
  const int num_buckets = 9;
  for (uint i=0; i<num_buckets; i++)
  {
    vector<float> r = compute_random_vector(seed+i, bucket_max_size_);
    r_.push_back(unit_vector(r));
  }
}

/** \brief Computes a random vector of some size
  * @return random vector
  * \param seed to generate the random values
  * \param desired size
  */
vector<float> haloc::Hash::compute_random_vector(uint seed, int size)
{
  srand(seed);
  vector<float> h;
  for (int i=0; i<size; i++)
    h.push_back( (float)rand()/RAND_MAX );
  return h;
}

/** \brief Make a vector unit
  * @return the "unitized" vector
  * \param the "non-unitized" vector
  */
vector<float> haloc::Hash::unit_vector(vector<float> x)
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

/** \brief Compute the hash matching between 2 hashes.
  * @return the distance
  * \param hash query
  * \param hash trainer
  */
float haloc::Hash::matching(Hash::HashBucket hash_q, Hash::HashBucket hash_t)
{
  // Get the mask matrix to enter to knnmatcher.
  // This is because buckets with a very different number of keypoints are not valid for matching.
  // Moreover, buckets with a small number of keypoints are rejected.
  uint q_size = hash_q.num_features.size();
  uint t_size = hash_t.num_features.size();
  int max_feat_num_q = *max_element(hash_q.num_features.begin(), hash_q.num_features.end());
  int max_feat_num_t = *max_element(hash_t.num_features.begin(), hash_t.num_features.end());
  int max_feat_num = max(max_feat_num_q, max_feat_num_t);
  Mat mask(q_size, t_size, CV_8UC1);
  for (uint n=0; n<q_size; n++)
  {
    int hash_q_feat = hash_q.num_features[n];
    for (uint m=0; m<t_size; m++)
    {
      int hash_c_feat = hash_t.num_features[m];

      // Check if number of features are coincident
      int max_feat = max(hash_q_feat, hash_c_feat);
      int min_feat = min(hash_q_feat, hash_c_feat);
      mask.at<uchar>(n, m) = ( (min_feat > max_feat*0.6) && (max_feat > max_feat_num*0.3) );
    }
  }

  // Brute force matching
  const int knn = 2;
  Ptr<DescriptorMatcher> descriptor_matcher;
  descriptor_matcher = DescriptorMatcher::create("BruteForce");
  vector<vector<DMatch> > knn_matches;
  descriptor_matcher->knnMatch(hash_q.hash_value, hash_t.hash_value, knn_matches, knn, mask);

  // Crosscheck threshold
  const float ratio = 0.7;
  vector<DMatch> matches;
  for (size_t m = 0; m < knn_matches.size(); m++)
  {
    // Sanity checks
    if (knn_matches[m].size() < 2) continue;
    if (knn_matches[m][0].queryIdx < 0 || knn_matches[m][0].queryIdx > q_size ||
        knn_matches[m][0].trainIdx < 0 || knn_matches[m][0].trainIdx > t_size) continue;

    // Check the mask
    bool match_allowed = mask.empty() ? true : mask.at<uchar>(
      knn_matches[m][0].queryIdx, knn_matches[m][0].trainIdx) > 0;
    if (match_allowed && knn_matches[m][0].distance <= knn_matches[m][1].distance * ratio)
    {
      matches.push_back(knn_matches[m][0]);
    }
  }

  // TODO: Normalize the number of matches and the distances. YOU KNOW THAT THE MAXIMUM NUMBER OF MATCHES IS 9!!!!

  return 0.0;
}
