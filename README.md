libhaloc
=============

ROS library for HAsh-based LOop Closure detection. This library provides the tools for loop closing detection based on image hashing. Image hashing consists of representing every image with a small vector (hash). Then the hash of image A can be compared with the hash of image B in a super fast way in order to determine if images are similar.

The image hashing implemented in this library is based on SIFT features so, since SIFT is invariant to scale and rotation, the image hashing will be also invariant to these properties.

The library works for both mono and stereo cameras and provides a transformation (2d for mono and 3d for stereo) when loop closures are found.

## Related paper

[Autonomous Robots][paper]

CITATION:
```bash
@Article{Negre Carrasco2015,
   author="Negre Carrasco, Pep Lluis
   and Bonin-Font, Francisco
   and Oliver-Codina, Gabriel",
   title="Global image signature for visual loop-closure detection",
   journal="Autonomous Robots",
   year="2015",
   pages="1--15",
   issn="1573-7527",
   doi="10.1007/s10514-015-9522-4",
   url="http://dx.doi.org/10.1007/s10514-015-9522-4"
}
```

## How to prepare your SLAM node


Modify your CMakeLists.txt as follows:

```bash
# Add this line before catkin_package()
find_package(libhaloc REQUIRED)

# Include the libhaloc directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${libhaloc_INCLUDE_DIRS}
  ...
  )

# Link your executable with libhaloc
target_link_libraries(your_executable_here
  ${catkin_LIBRARIES}
  ${libhaloc_LIBRARIES}
  ...
  )
```

Include the header in your .cpp file:
```bash
#include <libhaloc/lc.h>
```

## How to call the library


You can use libhaloc in two different ways:
- To generate the image hashes and then, use your own techniques to compare this hashes and retrieve the best candidates to close loop.
- To use the full power of libhaloc to search loop closing candidates and compute the homogeneous transformation (if any).

### To use libhaloc as the first option:

* Declare your haloc object:
```bash
haloc::Hash haloc_;
```

* Init your haloc object (only once):
```bash
if (!hash_.isInitialized())
   hash_.init(c_cluster_.getSift());
```

* Then, for every new image you process, extract its descriptor matrix (SIFT)
```bash
cv::Mat sift;
// Use opencv to extract the SIFT matrix and then:
vector<float> current_hash = hash_.getHash(sift);
```

* Now, you can store all the hashes into a table and compare it:
```bash
std::vector<float> matches;
for (uint i=0; i<my_table.size(); i++)
{
   float m = hash_.match(current_hash, my_table[i]);
   matches.push_back(matches);
}
```

* Finally, sort the vector of matches from smallest to largest. Take the first 1, 2, 3 or 4 smallest values as candidates to close loop.


### To use libhaloc as the second option:


1) Declare your haloc object (this object is different from the option 1!!!):
```bash
haloc::LoopClosure haloc_;
```

2) Set the haloc parameters and initialize (only once):
```bash
haloc::LoopClosure::Params params;
params.work_dir = whatever;
.
.
.
haloc_.setParams(params);
haloc_.init();
```

4) Then, for every image, call the function setNode() to store the image properties and getLoopClosure() to get any possible loop closure between the last image and any previous images.
```bash
// Mono version
int loop_closure_with; 	// <- Will contain the index of the image that closes loop with the last inserted (-1 if none).
tf::Transform trans;    // <- Will contain the transformation of the loop closure (if any).
haloc_.setNode(img);
bool valid = haloc_.getLoopClosure(loop_closure_with, trans);

// Stereo version
int loop_closure_with; 	// <- Will contain the index of the image that closes loop with the last inserted (-1 if none).
tf::Transform trans; 	// <- Will contain the transformation of the loop closure (if any).
haloc_.setNode(img_left, img_right);
bool valid = haloc_.getLoopClosure(loop_closure_with, trans);
```

In both cases, if valid is true, then a loop closure has been found.


## Example


Check [this][stereo_slam] integration for a 3D Stereo Slam.


## Most Important Parameters


* `work_dir` - Directory where the library will save the image information (must be writable!).
* `desc_matching_type` - Can be CROSSCHECK or RATIO.
* `desc_thresh_ratio` - Descriptor threshold for crosscheck matching (typically between 0.7-0.9) or ratio for ratio matching (typically between 0.6-0.8).
* `min_neighbor` - Minimum number of neighbors that will be skipped for the loop closure (typically between 5-20, but depends on the frame rate).
* `n_candidates` - Get the n first candidates of the hash matching (typically 2, 3, or 4).
* `min_matches` - Minimum number of descriptor matches to consider a matching as possible loop closure (>12).
* `min_inliers` - Minimum number of inliers to consider a matching as possible loop closure (>12).


[stereo_slam]: https://github.com/srv/stereo_slam
[paper]: http://link.springer.com/article/10.1007/s10514-015-9522-4
