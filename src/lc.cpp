#include "libhaloc/lc.h"
#include "libhaloc/utils.h"
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/distributions.hpp>
#include <cv.h>
#include <highgui.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace boost;
namespace fs=filesystem;

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
haloc::LoopClosure::Params::Params() :
  work_dir(""),
  image_dir(""),
  num_proj(DEFAULT_NUM_PROJ),
  desc_type("SIFT"),
  desc_matching_type("CROSSCHECK"),
  desc_thresh_ratio(DEFAULT_DESC_THRESH_RATIO),
  epipolar_thresh(DEFAULT_EPIPOLAR_THRESH),
  min_neighbor(DEFAULT_MIN_NEIGHBOR),
  n_candidates(DEFAULT_N_CANDIDATES),
  group_range(DEFAULT_GROUP_RANGE),
  min_matches(DEFAULT_MIN_MATCHES),
  min_inliers(DEFAULT_MIN_INLIERS),
  max_reproj_err(DEFAULT_MAX_REPROJ_ERR),
  verbose(DEFAULT_VERBOSE),
  save_images(DEFAULT_SAVE_IMAGES)
{}

/** \brief LoopClosure class constructor.
  */
haloc::LoopClosure::LoopClosure(){}

/** \brief Sets the class parameters.
  * \param stuct of parameters.
  */
void haloc::LoopClosure::setParams(const Params& params)
{
  params_ = params;

  // Will be the same
  params_.image_dir = params_.work_dir;

  // Log
  cout << "  work_dir           = " << params_.work_dir << endl;
  cout << "  image_dir          = " << params_.image_dir << endl;
  cout << "  num_proj           = " << params_.num_proj << endl;
  cout << "  desc_type          = " << params_.desc_type << endl;
  cout << "  desc_matching_type = " << params_.desc_matching_type << endl;
  cout << "  desc_thresh_ratio  = " << params_.desc_thresh_ratio << endl;
  cout << "  epipolar_thresh    = " << params_.epipolar_thresh << endl;
  cout << "  min_neighbor       = " << params_.min_neighbor << endl;
  cout << "  n_candidates       = " << params_.n_candidates << endl;
  cout << "  group_range        = " << params_.group_range << endl;
  cout << "  min_matches        = " << params_.min_matches << endl;
  cout << "  min_inliers        = " << params_.min_inliers << endl;
  cout << "  max_reproj_err     = " << params_.max_reproj_err << endl;
  cout << "  verbose            = " << params_.verbose << endl;
  cout << "  save_images        = " << params_.save_images << endl;

}

/** \brief Sets the camera model.
  * \param Stereo camera model.
  */
void haloc::LoopClosure::setCameraModel(image_geometry::StereoCameraModel stereo_camera_model,
                                        Mat camera_matrix)
{
  query_.setCameraModel(stereo_camera_model);
  camera_matrix_ = camera_matrix;
}

/** \brief Initializes the loop closure class.
  */
void haloc::LoopClosure::init()
{
  // Working directory sanity check
  if (params_.work_dir[params_.work_dir.length()-1] != '/')
    params_.work_dir += "/haloc_" + lexical_cast<string>(time(0));
  else
    params_.work_dir += "haloc_" + lexical_cast<string>(time(0));

  // Create the directory to store the keypoints and descriptors
  if (fs::is_directory(params_.work_dir))
    fs::remove_all(params_.work_dir);
  fs::path dir(params_.work_dir);
  if (!fs::create_directory(dir))
    ROS_ERROR("[Haloc:] ERROR -> Impossible to create the execution directory.");

  // Image directory sanity check
  if (params_.save_images)
  {
    if (params_.image_dir[params_.image_dir.length()-1] != '/')
      params_.image_dir += "/images/";
    else
      params_.image_dir += "images/";

    // Create the directory to store the keypoints and descriptors
    if (fs::is_directory(params_.image_dir))
      fs::remove_all(params_.image_dir);
    fs::path dir(params_.image_dir);
    if (!fs::create_directory(dir))
      ROS_ERROR("[Haloc:] ERROR -> Impossible to create the image directory.");
  }

  // Initialize image properties
  haloc::Image::Params img_params;
  img_params.desc_type = params_.desc_type;
  img_params.desc_matching_type = params_.desc_matching_type;
  img_params.desc_thresh_ratio = params_.desc_thresh_ratio;
  img_params.min_matches = params_.min_matches;
  img_params.epipolar_thresh = params_.epipolar_thresh;
  query_.setParams(img_params);

  // Init main variables
  hash_table_.clear();
  img_id_ = 0;
}

/** \brief Finalizes the loop closure class.
  */
void haloc::LoopClosure::finalize()
{
  // Log information
  if (params_.verbose && lc_candidate_positions_.size() > 0)
  {
    double sum = std::accumulate(lc_candidate_positions_.begin(), lc_candidate_positions_.end(), 0.0);
    double mean = sum / lc_candidate_positions_.size();
    ROS_INFO_STREAM("[libhaloc:] Mean loop closure candidate position: " << mean << "/" << params_.n_candidates << ".");
  }

  // Remove the temporal directory
  if (fs::is_directory(params_.work_dir))
    fs::remove_all(params_.work_dir);
}


/** \brief Compute kp, desc and hash for one image (mono version).
  * @return the node unique identifier
  * \param cvMat containing the image.
  */
int haloc::LoopClosure::setNode(Mat img)
{
  // Set the image
  if (!query_.setMono(img_id_, img))
    return -1;

  if (params_.save_images)
  {
    string path = params_.image_dir + lexical_cast<string>(img_id_) + ".png";
    imwrite(path, img);
  }

  // Initialize hash
  if (!hash_.isInitialized())
    hash_.init(query_.getDesc());

  // Save hash to table
  hash_table_.push_back(make_pair(query_.getId(), hash_.getHash(query_.getDesc())));

  // Save kp and descriptors
  vector<Point3f> empty;
  FileStorage fs(params_.work_dir+"/"+lexical_cast<string>(img_id_)+".yml", FileStorage::WRITE);
  write(fs, "id", query_.getId());
  write(fs, "kp", query_.getKp());
  write(fs, "desc", query_.getDesc());
  write(fs, "threed", empty);
  fs.release();
  img_id_++;

  return query_.getId();
}


/** \brief Compute kp, desc and hash for two images (stereo version).
  * @return the node unique identifier, -1 if the node cannot be inserted due to its poor quality.
  * \param cvMat containing the left image.
  * \param cvMat containing the right image.
  */
int haloc::LoopClosure::setNode(Mat img_l, Mat img_r)
{
  // Set the images
  if (!query_.setStereo(img_id_, img_l, img_r))
    return -1;

  if (params_.save_images)
  {
    string path = params_.image_dir + lexical_cast<string>(img_id_) + "_l.png";
    imwrite(path, img_l);
    path = params_.image_dir + lexical_cast<string>(img_id_) + "_r.png";
    imwrite(path, img_r);
  }

  // Initialize hash
  if (!hash_.isInitialized())
    hash_.init(query_.getDesc());

  // Save hash to table
  hash_table_.push_back(make_pair(query_.getId(), hash_.getHash(query_.getDesc())));

  // Save kp and descriptors
  FileStorage fs(params_.work_dir+"/"+lexical_cast<string>(img_id_)+".yml", FileStorage::WRITE);
  write(fs, "id", query_.getId());
  write(fs, "kp", query_.getKp());
  write(fs, "desc", query_.getDesc());
  write(fs, "threed", query_.get3D());
  fs.release();
  img_id_++;

  return query_.getId();
}

/** \brief Returns the hash for a some specific image id.
  * @return true if hash found
  * \param the image id.
  * \param the returning hash.
  */
bool haloc::LoopClosure::getHash(int img_id, vector<float>& hash)
{
  hash.clear();
  for (uint i=0; i<hash_table_.size(); i++)
  {
    if (hash_table_[i].first == img_id)
    {
      hash = hash_table_[i].second;
      return true;
    }
  }
  return false;
}

/** \brief Computes the hash matching between 2 images
  * @return true if hashes found
  * \param the image id a.
  * \param the image id b.
  * \param the returning hash matching.
  */
bool haloc::LoopClosure::hashMatching(int img_id_a, int img_id_b, float& matching)
{
  matching = -1.0;
  vector<float> hash_a, hash_b;
  if(!getHash(img_id_a, hash_a)) return false;
  if(!getHash(img_id_b, hash_b)) return false;
  matching = hash_.match(hash_a, hash_b);
  return true;
}

/** \brief Get the best n_candidates to close loop with the last image.
  * @return a hash matching vector containing the best image candidates and its likelihood.
  */
void haloc::LoopClosure::getCandidates(vector< pair<int,float> >& candidates)
{
  getCandidates(img_id_-1, candidates);
}

/** \brief Get the best n_candidates to close loop with the image specified by id.
  * @return a hash matching vector containing the best image candidates and its likelihood.
  * \param image id.
  */
void haloc::LoopClosure::getCandidates(int image_id,
                                       vector< pair<int,float> >& candidates)
{
  // Init
  candidates.clear();

  // Check if enough neighbors
  if ((int)hash_table_.size() <= params_.min_neighbor) return;

  // Query matching versus all the hash table
  vector< pair<int,float> > best_matchings;
  getBestMatchings(image_id, params_.n_candidates, best_matchings);

  // Build the likelihood vector
  vector< pair<int,float> > cur_likelihood;
  buildLikelihoodVector(best_matchings, cur_likelihood);

  // Merge current likelihood with the previous
  vector<float> matchings_likelihood;
  getMatchingsLikelihood(best_matchings, matchings_likelihood, cur_likelihood, prev_likelihood_);

  // Save for the next execution
  prev_likelihood_ = cur_likelihood;

  // Sanity check
  if (best_matchings.size() < 2) return;

  // Group similar images
  vector< vector<int> > groups;
  groupSimilarImages(best_matchings, groups);

  // Order matchings by likelihood
  vector< pair<int, float> > sorted_matchings;
  for (uint i=0; i<best_matchings.size(); i++)
    sorted_matchings.push_back(make_pair(best_matchings[i].first, matchings_likelihood[i]));
  sort(sorted_matchings.begin(), sorted_matchings.end(), haloc::Utils::sortByLikelihood);

  // Build the output
  for (uint i=0; i<groups.size(); i++)
  {
    int group_id = -1;
    float group_likelihood = 0.0;
    for (uint j=0; j<groups[i].size(); j++)
    {
      // Search this index into the matchings vector
      for (uint k=0; k<sorted_matchings.size(); k++)
      {
        if (groups[i][j] == sorted_matchings[k].first)
        {
          if (group_id < 0) group_id = groups[i][j];
          group_likelihood += sorted_matchings[k].second;
          break;
        }
      }
    }
    candidates.push_back(make_pair(group_id, group_likelihood));
  }

  // Sort candidates by likelihood
  sort(candidates.begin(), candidates.end(), haloc::Utils::sortByLikelihood);

}

/** \brief Try to find a loop closure between last node and all other nodes.
  * @return true if valid loop closure, false otherwise.
  * \param Return the index of the image that closes loop (-1 if no loop).
  */
bool haloc::LoopClosure::getLoopClosure(int& lc_img_id)
{
  tf::Transform trans;
  return getLoopClosure(lc_img_id, trans);
}

/** \brief Try to find a loop closure between last node and all other nodes.
  * @return true if valid loop closure, false otherwise.
  * \param Return the index of the image that closes loop (-1 if no loop).
  * \param Return the transform between nodes if loop closure is valid.
  */
bool haloc::LoopClosure::getLoopClosure(int& lc_img_id,
                                        tf::Transform& trans)
{
  // Get the candidates to close loop
  vector< pair<int,float> > hash_matching;
  getCandidates(hash_matching);
  if (hash_matching.size() == 0) return false;

  // Check for loop closure
  trans.setIdentity();
  lc_img_id = -1;
  int matches = 0;
  int max_matches = 0;
  int inliers = 0;
  int max_inliers = 0;
  bool valid = false;
  int final_img_idx = -1;
  string best_lc_found = "";
  for (uint i=0; i<hash_matching.size(); i++)
  {
    // Loop-closure?
    final_img_idx = i;
    string candidate_id = lexical_cast<string>(hash_matching[i].first);
    Image candidate = getImage(params_.work_dir+"/"+candidate_id+".yml");
    valid = compute(query_, candidate, matches, inliers, trans);

    // Log
    if (params_.verbose)
    {
      if (matches > max_matches)
      {
        max_matches = matches;
        max_inliers = inliers;
        best_lc_found = candidate_id;
      }
    }
    if (valid) break;
  }

  // Get the image of the loop closure
  if (valid)
  {
    // Store the loop closing information
    lc_img_id = hash_matching[final_img_idx].first;
    lc_candidate_positions_.push_back(final_img_idx+1);
    lc_found_.push_back(make_pair(lc_img_id, (img_id_ - 1) ));

    // Log
    if(params_.verbose)
      ROS_INFO_STREAM("[libhaloc:] LC by hash between nodes " <<
                      query_.getId() << " and " << lc_img_id <<
                      " (matches: " << matches << "; inliers: " <<
                      inliers << "; Position: " << final_img_idx+1 << "/" <<
                      params_.n_candidates << ").");
  }
  else
  {
    // Log
    if(params_.verbose && best_lc_found != "")
    {
      ROS_INFO_STREAM("[libhaloc:] No LC by hash, best candidate is " <<
                      best_lc_found << " (matches: " << max_matches <<
                      "; inliers: " << max_inliers << ").");
    }
  }

  // Return true if any valid loop closure has been found.
  return valid;
}

/** \brief Compute the loop closure (if any) between A -> B.
  * @return true if valid loop closure, false otherwise.
  * \param reference image id.
  * \param current image id.
  * \param Return the transform between nodes if loop closure is valid.
  * \param Show output log or not.
  * \param The number of matches
  * \param The number of inliers
  */
bool haloc::LoopClosure::getLoopClosure(int img_id_a,
                                        int img_id_b,
                                        tf::Transform& trans,
                                        int& matches,
                                        int& inliers,
                                        bool logging)
{
  // Image a
  string file_a = params_.work_dir+"/"+lexical_cast<string>(img_id_a)+".yml";
  Image img_a = getImage(file_a);

  // Image b
  string file_b = params_.work_dir+"/"+lexical_cast<string>(img_id_b)+".yml";
  Image img_b = getImage(file_b);

  // Get the loop closing (if any)
  bool valid = compute(img_a, img_b, matches, inliers, trans);

  // If valid loop closure, save it
  if (valid)
  {
    lc_found_.push_back(make_pair(img_id_a, img_id_b));
  }

  if(params_.verbose && logging)
  {
    if (valid)
    {
      ROS_INFO_STREAM("[libhaloc:] LC by ID between " <<
                      img_id_a << " and " << img_id_b <<
                      " (matches: " << matches << "; inliers: " <<
                      inliers << ").");
    }
    else
    {
      ROS_INFO_STREAM("[libhaloc:] No LC by ID between " <<
                      img_id_a << " and " << img_id_b <<
                      " (matches: " << matches << "; inliers: " <<
                      inliers << ").");
    }
  }
  return valid;
}

/** \brief Compute the loop closure (if any) between A -> B.
  * @return true if valid loop closure, false otherwise.
  * \param reference image id.
  * \param current image id.
  * \param Return the transform between nodes if loop closure is valid.
  * \param Show output log or not.
  */
bool haloc::LoopClosure::getLoopClosure(int img_id_a,
                                        int img_id_b,
                                        tf::Transform& trans,
                                        bool logging)
{
  int matches, inliers;
  return getLoopClosure(img_id_a, img_id_b, trans, matches, inliers, logging);
}

/** \brief Compute the loop closure (if any).
  * @return true if valid loop closure, false otherwise.
  * \param Query image object.
  * \param Candidate image object.
  * \param Return the number of matches found.
  * \param Return the number of inliers found.
  * \param Return the transform between nodes if loop closure is valid.
  */
bool haloc::LoopClosure::compute(Image query,
                                 Image candidate,
                                 int &matches,
                                 int &inliers,
                                 tf::Transform& trans)
{
  // Initialize outputs
  matches = 0;
  inliers = 0;

  // Descriptors matching
  Mat match_mask;
  vector<DMatch> desc_matches;
  if(params_.desc_matching_type == "CROSSCHECK")
  {
    haloc::Utils::crossCheckThresholdMatching(candidate.getDesc(),
        query.getDesc(), params_.desc_thresh_ratio, match_mask, desc_matches);
  }
  else if (params_.desc_matching_type == "RATIO")
  {
    haloc::Utils::ratioMatching(candidate.getDesc(),
        query.getDesc(), params_.desc_thresh_ratio, match_mask, desc_matches);
  }
  else
  {
    ROS_ERROR("[Haloc:] ERROR -> desc_matching_type must be 'CROSSCHECK' or 'RATIO'");
  }

  matches = (int)desc_matches.size();

  // Matches threshold
  if (matches < params_.min_matches)
    return false;

  // Get the matched keypoints
  vector<KeyPoint> query_kp = query.getKp();
  vector<KeyPoint> candidate_kp = candidate.getKp();
  vector<Point3f> candidate_3d = candidate.get3D();
  vector<Point2f> query_matched_kp;
  vector<Point2f> candidate_matched_kp;
  vector<Point3f> candidate_matched_3d_points;
  for(int i=0; i<matches; i++)
  {
    query_matched_kp.push_back(query_kp[desc_matches[i].trainIdx].pt);
    candidate_matched_kp.push_back(candidate_kp[desc_matches[i].queryIdx].pt);

    // Only stereo
    if (candidate_3d.size() != 0)
      candidate_matched_3d_points.push_back(candidate_3d[desc_matches[i].queryIdx]);
  }

  // Proceed depending on mono or stereo
  if (candidate_3d.size() == 0) // Mono
  {
    // Check the epipolar geometry
    Mat status;
    Mat F = findFundamentalMat(candidate_matched_kp, query_matched_kp, FM_RANSAC, params_.epipolar_thresh, 0.999, status);

    // Is the fundamental matrix valid?
    Scalar f_sum_parts = cv::sum(F);
    float f_sum = (float)f_sum_parts[0] + (float)f_sum_parts[1] + (float)f_sum_parts[2];
    if (f_sum < 1e-3)
      return false;

    // Check inliers size
    inliers = (int)cv::sum(status)[0];
    if (inliers < params_.min_inliers)
      return false;
  }
  else // Stereo
  {
    Mat rvec, tvec;
    vector<int> solvepnp_inliers;
    solvePnPRansac(candidate_matched_3d_points, query_matched_kp, camera_matrix_,
                   cv::Mat(), rvec, tvec, false,
                   100, params_.max_reproj_err,
                   40, solvepnp_inliers);

    // Inliers threshold
    inliers = (int)solvepnp_inliers.size();
    if (inliers < params_.min_inliers)
      return false;

    // Save inliers matches
    if (params_.save_images)
    {
      vector<DMatch> final_inliers;
      for(int i=0; i<solvepnp_inliers.size(); i++)
      {
        int idx = solvepnp_inliers[i];
        final_inliers.push_back(desc_matches[idx]);
      }
      Mat img_inliers;
      string path_query = params_.image_dir + lexical_cast<string>(query.getId()) + "_l.png";
      string path_candidate = params_.image_dir + lexical_cast<string>(candidate.getId()) + "_l.png";
      string output = params_.image_dir + "inliers_" + lexical_cast<string>(query.getId()) + "_" + lexical_cast<string>(candidate.getId()) + ".png";
      Mat img_query = imread(path_query, CV_LOAD_IMAGE_COLOR);
      Mat img_candidate = imread(path_candidate, CV_LOAD_IMAGE_COLOR);
      drawMatches(img_candidate, candidate_kp, img_query, query_kp, final_inliers, img_inliers);
      imwrite(output, img_inliers);
    }

    trans = haloc::Utils::buildTransformation(rvec, tvec);
  }

  // If we arrive here, there is a loop closure.
  return true;
}

/** \brief Retrieve the best n matchings give a certain image id.
  * \param Query image id.
  * \param Number of best matches to retrieve.
  * \param Stores best n matchings in a vector of pairs <image_id, distance>.
  */
void haloc::LoopClosure::getBestMatchings(int image_id,
                                          int best_n,
                                          vector< pair<int,float> > &best_matchings)
{
  // Create a list with the non-possible candidates (because they are already loop closings)
  vector<int> no_candidates;
  for (uint i=0; i<lc_found_.size(); i++)
  {
    if (lc_found_[i].first == image_id)
      no_candidates.push_back(lc_found_[i].second);
    if (lc_found_[i].second == image_id)
      no_candidates.push_back(lc_found_[i].first);
  }

  // Query hash
  vector<float> hash_q = hash_table_[image_id].second;

  // Loop over all the hashes stored
  vector< pair<int,float> > all_matchings;
  for (uint i=0; i<hash_table_.size()-params_.min_neighbor-1; i++)
  {
    // Do not compute the hash matching with itself
    if (hash_table_[i].first == image_id) continue;

    // Continue if candidate is in the no_candidates list
    if (find(no_candidates.begin(), no_candidates.end(), hash_table_[i].first) != no_candidates.end())
      continue;

    // Hash matching
    vector<float> hash_t = hash_table_[i].second;
    float m = hash_.match(hash_q, hash_t);
    all_matchings.push_back(make_pair(hash_table_[i].first, m));
  }

  // Sort the hash matchings
  sort(all_matchings.begin(), all_matchings.end(), haloc::Utils::sortByMatching);

  // Retrieve the best n matches
  best_matchings.clear();
  int max_size = best_n;
  if (max_size > all_matchings.size()) max_size = all_matchings.size();
  for (uint i=0; i<max_size; i++)
    best_matchings.push_back(all_matchings[i]);
}

/** \brief Compute the likelihood vectors given a certain hash matching set.
  * \param Hash matching vector in the format <image_id, distance>.
  * \param Stores the likelihood in the format <image_id, probability>.
  */
void haloc::LoopClosure::buildLikelihoodVector(vector< pair<int,float> > hash_matchings,
                                               vector< pair<int,float> > &likelihood)
{
  // Init
  likelihood.clear();

  // Get maximums and minimums of the hash matchings
  int min_idx = -1;
  int max_idx = -1;
  int min_hash = -1;
  int max_hash = -1;
  for (uint i=0; i<hash_matchings.size(); i++)
  {
    if (min_idx < 0 || hash_matchings[i].first < min_idx) min_idx = hash_matchings[i].first;
    if (max_idx < 0 || hash_matchings[i].first > max_idx) max_idx = hash_matchings[i].first;
    if (min_hash < 0 || hash_matchings[i].second < min_hash) min_hash = hash_matchings[i].second;
    if (max_hash < 0 || hash_matchings[i].second > max_hash) max_hash = hash_matchings[i].second;
  }

  // Normalize the hash values
  const float min_norm_val = 1.0;
  const float max_norm_val = 2.0;
  float m = (min_norm_val - max_norm_val) / (max_hash - min_hash);
  float n = max_norm_val - m*min_hash;

  // Build the probability vector
  int space = params_.group_range;
  for (int i=0; i<=(max_idx-min_idx)+2*space; i++)
  {
    int cur_idx = min_idx - space + i;

    // Compute the probability for every candidate
    float prob = 0.0;
    for (uint j=0; j<hash_matchings.size(); j++)
    {
      // Create the normal distribution for this matching
      math::normal_distribution<> nd((float)hash_matchings[j].first, 2.0);

      // Sanity check
      if (!isfinite(m))
        prob += min_norm_val * math::pdf(nd, (float)cur_idx);
      else
        prob += (m*hash_matchings[j].second + n) * math::pdf(nd, (float)cur_idx);
    }
    likelihood.push_back(make_pair(cur_idx,prob));
  }
}

/** \brief Given a vector of matches and the current and previous likelihood vectors, returns the
  * likelihood for these matches.
  * \param Hash matching vector in the format <image_id, distance>.
  * \param Stores the likelihood for the given matching vectors
  * \param Current vector of likelihood in the format <image_id, probability>.
  * \param Previous vector of likelihood in the format <image_id, probability>.
  */
void haloc::LoopClosure::getMatchingsLikelihood(vector< pair<int,float> > matchings,
                                                vector<float> &matchings_likelihood,
                                                vector< pair<int,float> > cur_likelihood,
                                                vector< pair<int,float> > prev_likelihood)
{
  // Init
  matchings_likelihood.clear();

  // Extract the vectors
  vector<int> cur_idx;
  for (uint i=0; i<cur_likelihood.size(); i++)
    cur_idx.push_back(cur_likelihood[i].first);

  vector<int> prev_idx;
  for (uint i=0; i<prev_likelihood.size(); i++)
    prev_idx.push_back(prev_likelihood[i].first);

  // For every matching
  for (uint i=0; i<matchings.size(); i++)
  {
    // Find previous value
    float prev_prob = 0.0;
    vector<int>::iterator itp = find(prev_idx.begin(), prev_idx.end(), matchings[i].first);
    if (itp != prev_idx.end())
    {
      int idx = distance(prev_idx.begin(), itp);
      prev_prob = prev_likelihood[idx].second;
    }

    // Find current value
    float cur_prob = 0.0;
    vector<int>::iterator itc = find(cur_idx.begin(), cur_idx.end(), matchings[i].first);
    if (itc != cur_idx.end())
    {
      int idx = distance(cur_idx.begin(), itc);
      cur_prob = cur_likelihood[idx].second;
    }

    // Add and save
    matchings_likelihood.push_back(prev_prob + cur_prob);
  }

  // Make the probability of sum = 1
  float x_norm = 0.0;
  for (uint i=0; i<matchings_likelihood.size(); i++)
    x_norm += fabs(matchings_likelihood[i]);
  for (uint i=0; i<matchings_likelihood.size(); i++)
    matchings_likelihood[i] = matchings_likelihood[i] / x_norm;
}

/** \brief Group the matchings by images with similar id
  * \param Hash matching vector in the format <image_id, distance>.
  * \param Stores groups of images
  */
void haloc::LoopClosure::groupSimilarImages(vector< pair<int,float> > matchings,
                                            vector< vector<int> > &groups)
{
  // Init groups vector
  groups.clear();
  vector<int> new_group;
  new_group.push_back(matchings[0].first);
  groups.push_back(new_group);
  matchings.erase(matchings.begin());

  bool finish = false;
  while(!finish)
  {
    // Get the last inserted group
    vector<int> last_group = groups.back();

    // Mean image index
    int sum = accumulate(last_group.begin(), last_group.end(), 0.0);
    float mean = (float)sum / (float)last_group.size();

    bool new_insertion = false;
    for (uint i=0; i<matchings.size(); i++)
    {
      if ( abs(mean - (float)matchings[i].first) < params_.group_range )
      {
        // Save id
        last_group.push_back(matchings[i].first);

        // Replace group
        groups.pop_back();
        groups.push_back(last_group);

        // Delete from matching list
        matchings.erase(matchings.begin() + i);

        new_insertion = true;
        break;
      }
    }

    // Finish?
    if (matchings.size() == 0)
    {
      finish = true;
      continue;
    }

    // Proceed depending on new insertion or not
    if (!new_insertion)
    {
      new_group.clear();
      new_group.push_back(matchings[0].first);
      groups.push_back(new_group);
      matchings.erase(matchings.begin());
    }
  }
}


/** \brief Get the image information from file.
  * @return the image object.
  * \param image filename
  */
haloc::Image haloc::LoopClosure::getImage(string img_file)
{
  // Init
  Image img;

  // Sanity check
  if ( !fs::exists(img_file) ) return img;

  // Get the image keypoints and descriptors
  FileStorage fs;
  fs.open(img_file, FileStorage::READ);
  if (!fs.isOpened())
    ROS_ERROR("[Haloc:] ERROR -> Failed to open the image keypoints and descriptors.");
  int id;
  vector<KeyPoint> kp;
  Mat desc;
  vector<Point3f> p3d;
  fs["id"] >> id;
  fs["desc"] >> desc;
  fs["threed"] >> p3d;
  FileNode kp_node = fs["kp"];
  read(kp_node, kp);
  fs.release();

  // Set the properties of the image
  img.setId(id);
  img.setKp(kp);
  img.setDesc(desc);
  img.set3D(p3d);

  return img;
}