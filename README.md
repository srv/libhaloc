libhaloc
=============

ROS library for HAsh-based LOop Closure. This library provides the tools for image loop closing based on image hashing. Image hashing consists of representing every image with a small vector (hash). Then the hash of image A can be compared with the hash of image B in a super fast way in order to determine if images are similar.

The image hashing implemented in this library is based on features (SIFT, SURF, ORB, etc.) so, if the features you choose are invariant to scale and rotation, the image hashing will be also invariant to these properties.

The library works for both mono and stereo datasets and provides a transformation (2d for mono and 3d for stereo) when loop closures are found.

How it works
-------
1) Declare your haloc object like this:
```bash
haloc::LoopClosure haloc_obj;
```

2) Set the haloc parameters:
```bash
haloc::LoopClosure::Params params;
params.work_dir = whatever;
params.desc_type = whatever;
.
.
.
params.validate = whatever;
haloc_obj.setParams(params);
```

3) Initialize the object
```bash
haloc_obj.init();
```

4) Then, for every image, call to setNode to save the new image and getLoopClosure to get any possible loop closure between the last image and any previous images.
```bash
// Mono version
int loop_closure_with; 	// <- Will contain the index of the image that closes loop with the last inserted (-1 if none).
haloc_obj.setNode(img);
bool valid = haloc_obj.getLoopClosure(loop_closure_with);

// Stereo version
int loop_closure_with; 	// <- Will contain the index of the image that closes loop with the last inserted (-1 if none).
tf::Transform trans; 	// <- Will contain the transformation of the loop closure (if any).
haloc_obj.setNode(img_left, img_right);
bool valid = haloc_obj.getLoopClosure(loop_closure_with, trans);
```

In both cases, if valid is true, then a loop closure has been found, false otherwise.


Parameters
-------

* `work_dir` - Directory where the library will save the image informations (must be writtible!).
* `desc_type` - Type of the descriptors (can be SIFT, SURF).
* `num_proj` -  Number of projections for image hashing (1-5).
* `desc_thresh` - Descriptor threshold (tipically between 0.7-0.9).
* `epipolar_thresh` - Epipolar threshold.
* `min_neighbour` - Minimum number of neighbours that will be skiped for the loop closure (tipically between 5-20, but depends on the frame rate).
* `n_candidates` - Get the n first candidates of the hash matching (tipically between 1-5).
* `min_matches` - Minimun number of descriptor matches to consider a matching as possible loop closure (>8).
* `min_inliers` - Minimum number of inliers to consider a matching as possible loop closure (>8).
* `max_reproj_err` - Maximum reprojection error (stereo only).
* `validate` - True if you want to validate the loop closure (spends more time). Default False.

Demo
-------

See it in action [here][link_demo].


[link_demo]: http://pul.uib.es/libhaloc/
