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

namespace fs=boost::filesystem;

/** \brief Parameter constructor. Sets the parameter struct to default values.
  */
haloc::LoopClosure::Params::Params() :
  num_proj(DEFAULT_NUM_PROJ),
  work_dir(""),
  desc_type("SIFT"),
  desc_matching_type("CROSSCHECK"),
  max_keypoints(DEFAULT_MAX_KEYPOINTS),
  desc_thresh_ratio(DEFAULT_DESC_THRESH_RATIO),
  epipolar_thresh(DEFAULT_EPIPOLAR_THRESH),
  min_neighbour(DEFAULT_MIN_NEIGHBOUR),
  n_candidates(DEFAULT_N_CANDIDATES),
  min_matches(DEFAULT_MIN_MATCHES),
  min_inliers(DEFAULT_MIN_INLIERS),
  max_reproj_err(DEFAULT_MAX_REPROJ_ERR),
  validate(DEFAULT_VALIDATE),
  verbose(DEFAULT_VERBOSE)
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

  // Log
  cout << "  num_proj           = " << params_.num_proj << endl;
  cout << "  work_dir           = " << params_.work_dir << endl;
  cout << "  desc_type          = " << params_.desc_type << endl;
  cout << "  desc_matching_type = " << params_.desc_matching_type << endl;
  cout << "  max_keypoints      = " << params_.max_keypoints << endl;
  cout << "  desc_thresh_ratio  = " << params_.desc_thresh_ratio << endl;
  cout << "  epipolar_thresh    = " << params_.epipolar_thresh << endl;
  cout << "  min_neighbour      = " << params_.min_neighbour << endl;
  cout << "  n_candidates       = " << params_.n_candidates << endl;
  cout << "  min_matches        = " << params_.min_matches << endl;
  cout << "  min_inliers        = " << params_.min_inliers << endl;
  cout << "  max_reproj_err     = " << params_.max_reproj_err << endl;
  cout << "  validate           = " << params_.validate << endl;
  cout << "  verbose            = " << params_.verbose << endl;

}

/** \brief Sets the camera model.
  * \param Stereo camera model.
  */
void haloc::LoopClosure::setCameraModel(image_geometry::StereoCameraModel stereo_camera_model, Mat camera_matrix)
{
  img_.setCameraModel(stereo_camera_model);
  camera_matrix_ = camera_matrix;
}

/** \brief Initializes the loop closure class.
  */
void haloc::LoopClosure::init()
{
  // Working directory sanity check
  if (params_.work_dir[params_.work_dir.length()-1] != '/')
    params_.work_dir += "/haloc_" + boost::lexical_cast<string>(time(0));
  else
    params_.work_dir += "haloc_" + boost::lexical_cast<string>(time(0));

  // Create the directory to store the keypoints and descriptors
  if (fs::is_directory(params_.work_dir))
    fs::remove_all(params_.work_dir);
  fs::path dir(params_.work_dir);
  if (!fs::create_directory(dir))
    ROS_ERROR("[Haloc:] ERROR -> Impossible to create the execution directory.");

  // Initialize image properties
  haloc::Image::Params img_params;
  img_params.desc_type = params_.desc_type;
  img_params.desc_matching_type = params_.desc_matching_type;
  img_params.max_keypoints = params_.max_keypoints;
  img_params.desc_thresh_ratio = params_.desc_thresh_ratio;
  img_params.min_matches = params_.min_matches;
  img_params.epipolar_thresh = params_.epipolar_thresh;
  img_.setParams(img_params);

  // Initialize hash
  haloc::Hash::Params hash_params;
  hash_params.num_proj = params_.num_proj;
  hash_.setParams(hash_params);

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
  * @return true if valid node, false otherwise.
  * \param cvMat containing the image.
  */
bool haloc::LoopClosure::setNode(Mat img)
{
  string img_name = boost::lexical_cast<string>(img_id_);
  return setNode(img, img_name);
}

/** \brief Compute kp, desc and hash for one image (mono version).
  * @return true if valid node, false otherwise.
  * \param cvMat containing the image.
  * \param human readable name for this node.
  */
bool haloc::LoopClosure::setNode(Mat img, string name)
{
  // Set the image
  if(!img_.setMono(img, name)) return false;

  // Initialize hash
  if (!hash_.isInitialized())
    hash_.init(img_.getDesc(), true);

  // Save hash to table
  hash_table_.push_back(make_pair(img_id_, hash_.getHash(img_.getDesc())));

  // Save kp and descriptors
  vector<Point3f> empty;
  FileStorage fs(params_.work_dir+"/"+boost::lexical_cast<string>(img_id_)+".yml", FileStorage::WRITE);
  write(fs, "id", img_id_);
  write(fs, "name", name);
  write(fs, "kp", img_.getKp());
  write(fs, "desc", img_.getDesc());
  write(fs, "threed", empty);
  fs.release();
  img_id_++;

  return true;
}

/** \brief Compute kp, desc and hash for two images (stereo version).
  * @return true if valid node, false otherwise.
  * @return the assigned node name.
  * \param cvMat containing the left image.
  * \param cvMat containing the right image.
  */
bool haloc::LoopClosure::setNode(Mat img_l, Mat img_r)
{
  string img_name = boost::lexical_cast<string>(img_id_);
  return setNode(img_l, img_r, img_name);
}

/** \brief Compute kp, desc and hash for two images (stereo version).
  * @return true if valid node, false otherwise.
  * \param cvMat containing the left image.
  * \param cvMat containing the right image.
  * \param human readable name for this node.
  */
bool haloc::LoopClosure::setNode(Mat img_l, Mat img_r, string name)
{
  // Set the image
  if(!img_.setStereo(img_l, img_r, name)) return false;

  // Initialize hash
  if (!hash_.isInitialized())
    hash_.init(img_.getDesc(), true);

  // Save hash to table
  hash_table_.push_back(make_pair(img_id_, hash_.getHash(img_.getDesc())));

  // Save kp and descriptors
  FileStorage fs(params_.work_dir+"/"+boost::lexical_cast<string>(img_id_)+".yml", FileStorage::WRITE);
  write(fs, "id", img_id_);
  write(fs, "name", name);
  write(fs, "kp", img_.getKp());
  write(fs, "desc", img_.getDesc());
  write(fs, "threed", img_.get3D());
  fs.release();
  img_id_++;

  return true;
}

/** \brief Get the best n_candidates to close loop with the last image.
  * @return a hash matching vector containing the best image matchings and its likelihood.
  */
void haloc::LoopClosure::getCandidates(vector< pair<int,float> >& matchings)
{
  getCandidates(img_id_-1, matchings);
}

/** \brief Get the best n_candidates to close loop with the image specified by id.
  * \param image id.
  */
void haloc::LoopClosure::getCandidates(int image_id, vector< pair<int,float> >& matchings)
{
  // Check if enough neighbours
  if (hash_table_.size() - 1 <= params_.min_neighbour || hash_table_.size() == 0) return;

  // The query hash
  vector<float> hash_q = hash_table_[image_id].second;

  // Loop over all the hashes stored
  vector< pair<int,float> > tmp_matchings;
  for (uint i=0; i<hash_table_.size()-params_.min_neighbour-1; i++)
  {
    // Do not compute the hash matching with itself
    if (i == image_id) continue;

    vector<float> hash_t = hash_table_[i].second;
    float m = hash_.match(hash_q, hash_t);
    tmp_matchings.push_back(make_pair(hash_table_[i].first, m));
  }

  // Sort the hash matchings
  sort(tmp_matchings.begin(), tmp_matchings.end(), haloc::Utils::sortByMatching);

  // Retrieve the first n_candidates
  matchings.clear();
  int max_size = params_.n_candidates;
  if (max_size > tmp_matchings.size()) max_size = tmp_matchings.size();
  int min_idx = -1;
  int max_idx = -1;
  int min_hash = -1;
  int max_hash = -1;
  for (uint i=0; i<max_size; i++)
  {
    if (min_idx < 0 || tmp_matchings[i].first < min_idx) min_idx = tmp_matchings[i].first;
    if (max_idx < 0 || tmp_matchings[i].first > max_idx) max_idx = tmp_matchings[i].first;
    if (min_hash < 0 || tmp_matchings[i].second < min_hash) min_hash = tmp_matchings[i].second;
    if (max_hash < 0 || tmp_matchings[i].second > max_hash) max_hash = tmp_matchings[i].second;

    matchings.push_back(tmp_matchings[i]);
  }

  // Normalize the hash values
  const float min_norm_val = 1.0;
  const float max_norm_val = 2.0;
  float m = (min_norm_val - max_norm_val) / (max_hash - min_hash);
  float n = max_norm_val - m*min_hash;

  // Build the probability vector
  const int space = 10;
  vector<int> prob_idx;
  vector<float> prob_val;
  for (int i=0; i<=(max_idx-min_idx)+2*space; i++)
  {
    int cur_idx = min_idx - space + i;
    prob_idx.push_back(cur_idx);

    // Compute the probability for every candidate
    float prob = 0.0;
    for (uint j=0; j<matchings.size(); j++)
    {
      // Create the normal distribution for this matching
      boost::math::normal_distribution<> nd((float)matchings[j].first,2.0);
      prob += (m*matchings[j].second + n) * boost::math::pdf(nd, (float)cur_idx);
    }
    prob_val.push_back(prob);
  }

  // Merge current probability with the previous
  vector<float> matchings_prob;
  for (uint i=0; i<matchings.size(); i++)
  {
    // Find previous value
    float prev_prob = 0.0;
    vector<int>::iterator itp = find(prob_idx_.begin(), prob_idx_.end(), matchings[i].first);
    if (itp != prob_idx_.end())
    {
      int idx = distance(prob_idx_.begin(), itp);
      prev_prob = prob_val_[idx];
    }

    // Find current value
    float cur_prob = 0.0;
    vector<int>::iterator itc = find(prob_idx.begin(), prob_idx.end(), matchings[i].first);
    if (itc != prob_idx.end())
    {
      int idx = distance(prob_idx.begin(), itc);
      cur_prob = prob_val[idx];
    }

    // Add and save
    matchings_prob.push_back(prev_prob + cur_prob);
  }

  // Save for the next execution
  prob_idx_ = prob_idx;
  prob_val_ = prob_val;

  // Order matchings by probability
  vector< pair<int, float> > to_sort;
  for (uint i=0; i<matchings.size(); i++)
    to_sort.push_back(make_pair(matchings[i].first, matchings_prob[i]));

  sort(to_sort.begin(), to_sort.end(), haloc::Utils::sortByProbability);
  matchings = to_sort;
}

/** \brief Try to find a loop closure between last node and all other nodes.
  * @return true if valid loop closure, false otherwise.
  * \param Return the index of the image that closes loop (-1 if no loop).
  */
bool haloc::LoopClosure::getLoopClosure(int& lc_img_id, string& lc_name)
{
  tf::Transform trans;
  return getLoopClosure(lc_img_id, lc_name, trans);
}

/** \brief Try to find a loop closure between last node and all other nodes.
  * @return true if valid loop closure, false otherwise.
  * \param Return the index of the image that closes loop (-1 if no loop).
  * \param Return the name of the image that closes loop (empty if no loop).
  * \param Return the transform between nodes if loop closure is valid.
  */
bool haloc::LoopClosure::getLoopClosure(int& lc_img_id, string& lc_name, tf::Transform& trans)
{
  // Get the candidates to close loop
  vector< pair<int,float> > matchings;
  getCandidates(matchings);
  if (matchings.size() == 0) return false;

  // Check for loop closure
  trans.setIdentity();
  lc_img_id = -1;
  lc_name = "";
  int matches = 0;
  int max_matches = 0;
  int inliers = 0;
  int max_inliers = 0;
  bool valid = false;
  string best_lc_found = "";
  int valid_idx = 0;
  for (uint i=0; i<matchings.size(); i++)
  {
    // Store the index
    valid_idx = i;

    // Loop-closure?
    valid = compute(img_,
                    params_.work_dir+"/"+boost::lexical_cast<string>(matchings[i].first)+".yml",
                    lc_name,
                    matches,
                    inliers,
                    trans);

    // Log
    if (params_.verbose)
    {
      if (matches > max_matches)
      {
        max_matches = matches;
        max_inliers = inliers;
        best_lc_found = lc_name;
      }
    }

    // If the loop closure is valid and the seconds step validation is disabled, that's all.
    if (valid && !params_.validate) break;

    // Validate the loop closure?
    if (valid && params_.validate)
    {
      // Initialize validation
      bool validate_valid = false;
      int matches_val, inliers_val;
      tf::Transform trans_val;
      string tmp_name;

      // Loop closure for the previous image?
      validate_valid = compute(img_,
                               params_.work_dir+"/"+boost::lexical_cast<string>(matchings[i].first - 1)+".yml",
                               tmp_name,
                               matches_val,
                               inliers_val,
                               trans_val);

      if (!validate_valid)
      {
        // Previous validation does not works, try to validate with the next image
        validate_valid = compute(img_,
                                 params_.work_dir+"/"+boost::lexical_cast<string>(matchings[i].first + 1)+".yml",
                                 tmp_name,
                                 matches_val,
                                 inliers_val,
                                 trans_val);
      }

      // If validation, exit. If not, mark as non-valid
      if (validate_valid)
        break;
      else
        valid = false;
    }
  }

  // Get the image of the loop closure
  if (valid && valid_idx < matchings.size())
  {
    lc_img_id = matchings[valid_idx].first;
    lc_candidate_positions_.push_back(valid_idx);

    // Log
    if(params_.verbose)
      ROS_INFO_STREAM("[libhaloc:] LC between nodes " <<
                      img_.getName() << " and " << lc_name <<
                      " (matches: " << matches << "; inliers: " <<
                      inliers << "; Position: " << valid_idx << "/" <<
                      params_.n_candidates << ").");
  }
  else
  {
    // Log
    if(params_.verbose && best_lc_found != "")
    {
      ROS_INFO_STREAM("[libhaloc:] No LC, but best candidate is node " <<
                      best_lc_found << " (matches: " << max_matches <<
                      "; inliers: " << max_inliers << ").");
    }
    lc_name = "";
  }

  // Return true if any valid loop closure has been found.
  return valid;
}

/** \brief Compute the loop closure (if any) between ref -> cur.
  * @return true if valid loop closure, false otherwise.
  * \param reference image id.
  * \param current image id.
  * \param Return the number of matches found.
  * \param Return the number of inliers found.
  * \param Return the transform between nodes if loop closure is valid.
  */
bool haloc::LoopClosure::getLoopClosure(string image_id_ref,
                                        string image_id_cur,
                                        tf::Transform& trans)
{
  // Read the data for image_id_1
  FileStorage fs;
  fs.open(params_.work_dir+"/"+image_id_ref+".yml", FileStorage::READ);
  if (!fs.isOpened())
    ROS_ERROR("[Haloc:] ERROR -> Failed to open the image keypoints and descriptors.");
  string name;
  vector<Point2f> kp;
  Mat desc;
  vector<Point3f> points_3d;
  fs["name"] >> name;
  fs["kp"] >> kp;
  fs["desc"] >> desc;
  fs["threed"] >> points_3d;
  fs.release();
  Image img_ref;
  img_ref.setName(name);
  img_ref.setKp(kp);
  img_ref.setDesc(desc);
  img_ref.set3D(points_3d);

  // Get the loop closing (if any)
  int matches, inliers;
  bool valid = compute(img_ref,
                       params_.work_dir+"/"+image_id_cur+".yml",
                       name,
                       matches,
                       inliers,
                       trans);

  if(params_.verbose && valid)
    ROS_INFO_STREAM("[libhaloc:] Loop closed by ID between " <<
                    image_id_ref << " and " << image_id_cur <<
                    " (matches: " << matches << "; inliers: " <<
                    inliers << ").");
  return valid;
}

/** \brief Compute the loop closure (if any).
  * @return true if valid loop closure, false otherwise.
  * \param Reference image object.
  * \param Current image filename with all the properties.
  * \param Return the number of matches found.
  * \param Return the number of inliers found.
  * \param Return the transform between nodes if loop closure is valid.
  */
bool haloc::LoopClosure::compute(Image ref_image,
                                 string cur_filename,
                                 string &lc_name,
                                 int &matches,
                                 int &inliers,
                                 tf::Transform& trans)
{
  // Initialize outputs
  matches = 0;
  inliers = 0;

  // Sanity check
  if ( !fs::exists(cur_filename) ) return false;

  // Get the image keypoints and descriptors
  FileStorage fs;
  fs.open(cur_filename, FileStorage::READ);
  if (!fs.isOpened())
    ROS_ERROR("[Haloc:] ERROR -> Failed to open the image keypoints and descriptors.");
  vector<Point2f> cur_kp;
  Mat cur_desc;
  vector<Point3f> cur_3d;
  fs["name"] >> lc_name;
  fs["kp"] >> cur_kp;
  fs["desc"] >> cur_desc;
  fs["threed"] >> cur_3d;
  fs.release();

  // Descriptors matching
  Mat match_mask;
  vector<DMatch> desc_matches;
  if(params_.desc_matching_type == "CROSSCHECK")
  {
    haloc::Utils::crossCheckThresholdMatching(cur_desc,
        ref_image.getDesc(), params_.desc_thresh_ratio, match_mask, desc_matches);
  }
  else if (params_.desc_matching_type == "RATIO")
  {
    haloc::Utils::ratioMatching(cur_desc,
        ref_image.getDesc(), params_.desc_thresh_ratio, desc_matches);
  }
  else
  {
    ROS_ERROR("[Haloc:] ERROR -> desc_matching_type must be 'CROSSCHECK' or 'RATIO'");
  }

  matches = (int)desc_matches.size();

  // Check matches size
  if (matches < params_.min_matches)
    return false;

  // Get the matched keypoints
  vector<Point2f> ref_kp = ref_image.getKp();
  vector<Point2f> ref_matched_kp;
  vector<Point2f> cur_matched_kp;
  vector<Point3f> cur_matched_3d_points;
  for(int i=0; i<matches; i++)
  {
    ref_matched_kp.push_back(ref_kp[desc_matches[i].trainIdx]);
    cur_matched_kp.push_back(cur_kp[desc_matches[i].queryIdx]);

    // Only stereo
    if (cur_3d.size() != 0)
      cur_matched_3d_points.push_back(cur_3d[desc_matches[i].queryIdx]);
  }

  // Proceed depending on mono or stereo
  if (cur_3d.size() == 0) // Mono
  {
    // Check the epipolar geometry
    Mat status;
    Mat F = findFundamentalMat(cur_matched_kp, ref_matched_kp, FM_RANSAC, params_.epipolar_thresh, 0.999, status);

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
    solvePnPRansac(cur_matched_3d_points, ref_matched_kp, camera_matrix_,
                   cv::Mat(), rvec, tvec, false,
                   100, params_.max_reproj_err,
                   40, solvepnp_inliers);

    inliers = (int)solvepnp_inliers.size();
    if (inliers < params_.min_inliers)
      return false;

    trans = haloc::Utils::buildTransformation(rvec, tvec);
  }

  // If we arrive here, there is a loop closure.
  return true;
}