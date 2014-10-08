#ifndef HASH_H
#define HASH_H

#include <ros/ros.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <Eigen/Eigen>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace haloc
{

class Hash
{

public:

  // Represents a hash for a single bucket
  struct HashBucket
  {
    // Class parameters
    vector<int> num_features;           //!> Number of features per bucket.
    Mat hash_value;                     //!> Hash values per bucket.
  };

  // Class constructor
  Hash();

  // Returns true if the class has been initialized
  bool isInitialized();

  // Initialize class
  void init();

  // Compute the hash
  HashBucket getHash(vector<Mat> desc);

  // Compute the distance between 2 hashes
  float matching(HashBucket hash_q, HashBucket hash_t);

private:

  // Init the random vectors for projections
  void initProjections();

  // Compute a random vector
  vector<float> compute_random_vector(uint seed, int size);

  // Make a vector unit
  vector<float> unit_vector(vector<float> x);

  // Properties
  vector< vector<float> > r_;               //!> Vector of random values.
  bool initialized_;                        //!> True when class has been initialized.
  int desc_length_;                         //!> The length of the descriptors used.


  const int bucket_max_size_;               //!> The size of the descriptors may vary... We take a value large enough.
};

} // namespace

#endif // HASH_H