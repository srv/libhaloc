/* All openfabmap related functions thanks to https://code.google.com/p/cyphy-vis-slam/
 */

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <numeric>
#include "libhaloc/lc.h"

// vlfeat library
#include <kmeans.h>
#include <host.h>
#include <kdtree.h>
#include <vlad.h>
#include <vector>


namespace fs=boost::filesystem;

using namespace std;
using namespace cv;

class EvaluationNode
{

  public:

    // Public parameters
    string training_dir_;
    string running_dir_;
    haloc::LoopClosure lc_;
    string output_file_;

    // Class constructor
    EvaluationNode() : nhp_("~") {}

    // Read node parameters
    void readParameters()
    {
      // Directories
      nhp_.param("training_dir", training_dir_, string(""));
      nhp_.param("running_dir", running_dir_, string(""));
      nhp_.param<string>("vocab_path", vocab_path_, "vocab.yml");
      nhp_.param<string>("cl_tree_path", cl_tree_path_, "clTree.yml");
      nhp_.param<string>("trainbows_path", trainbows_path_, "trainbows.yml");
      nhp_.param<string>("gt_file", gt_file_, std::string(""));

      // BoW training parameters
      nhp_.param<int>("max_images", max_images_, 90);
      nhp_.param<double>("cluster_size", cluster_size_, 320.0);
      nhp_.param<double>("lower_information_bound", lower_information_bound_, 0.0);
      nhp_.param<int>("min_descriptor_count", min_descriptor_count_, 40);

      // BoW run parameters
      nhp_.param<int>("max_matches", max_matches_, 0);
      nhp_.param<double>("min_match_value", min_match_value_, 0.0);
      nhp_.param<bool>("disable_self_match", disable_self_match_, false);
      nhp_.param<int>("self_match_window", self_match_window_, 1);
      nhp_.param<bool>("disable_unknown_match", disable_unknown_match_, false);
      nhp_.param<bool>("add_only_new_places", add_only_new_places_, false);

      // VLAD parameters
      int dimension, num_centers, maxiter, maxcomp, maxrep, ntrees;
      nhp_.param<int>("dimension", dimension, 128);
      nhp_.param<int>("num_centers", num_centers, 64);
      nhp_.param<int>("maxiter", maxiter, 10);
      nhp_.param<int>("maxrep", maxrep, 1);
      nhp_.param<int>("ntrees", ntrees, 1);
      nhp_.param<int>("maxcomp", maxcomp, 100);
      dimension_ = dimension;
      num_centers_ = num_centers;
      maxiter_ = maxiter;
      maxrep_ = maxrep;
      ntrees_ = ntrees;
      maxcomp_ = maxcomp;

      // Libhaloc
      haloc::LoopClosure::Params lc_params;
      nhp_.param("work_dir", lc_params.work_dir, string(""));
      nhp_.param("desc_type", lc_params.desc_type, string("SIFT"));
      nhp_.param("desc_matching_type", lc_params.desc_matching_type, string("CROSSCHECK"));
      nhp_.param<double>("desc_thresh_ratio", lc_params.desc_thresh_ratio, 0.7);
      nhp_.param<int>("num_proj", lc_params.num_proj, 2);
      nhp_.param<int>("min_neighbor", lc_params.min_neighbor, 1);
      nhp_.param<int>("n_candidates", lc_params.n_candidates, 10);
      min_neighbor_ = lc_params.min_neighbor;
      n_candidates_ = lc_params.n_candidates;
      lc_.setParams(lc_params);

      // Output file
      output_file_ = lc_params.work_dir + "/output.txt";
      remove(output_file_.c_str());
    }

    // Initialize the node
    void init()
    {
      // Feature tools
      detector_ = new SIFT();
      extractor_ = new SIFT();
      matcher_ = new FlannBasedMatcher();
      bide_ = new BOWImgDescriptorExtractor(extractor_, matcher_);

      // BoW trainer
      trainer_ = of2::BOWMSCTrainer(cluster_size_);

      // Initialize for the first to contain
      // - Match to current
      // - Match to nothing
      confusion_mat_ = Mat::zeros(2,2,CV_64FC1);

      // Libhaloc
      lc_.init();
    }

    // Openfabmap learning stage
    void training()
    {
      // Sort directory of images
      typedef std::vector<fs::path> vec;
      vec v;
      copy(
        fs::directory_iterator(training_dir_),
        fs::directory_iterator(),
        back_inserter(v)
      );
      sort(v.begin(), v.end());
      vec::const_iterator it(v.begin());

      // Store all the descriptors to compute the cluster size
      Mat all_descriptors;

      // Iterate over all images
      int train_count = 0;
      while (it!=v.end())
      {
        if (fs::is_directory(*it))
        {
          // Next directory entry and continue
          it++;
          continue;
        }

        // Read image
        string filename = it->filename().string();
        string path = training_dir_ + "/" + filename;
        Mat img = imread(path, CV_LOAD_IMAGE_COLOR);

        ROS_INFO("[Haloc:] Detect.");
        vector<KeyPoint> kpts;
        detector_->detect(img, kpts);
        ROS_INFO("[Haloc:] Extract.");
        Mat descriptors;
        extractor_->compute(img, kpts, descriptors);

        // Store the descriptors
        for (int i = 1; i < descriptors.rows; i++)
        {
          all_descriptors.push_back(descriptors.row(i));
        }

        // Check if frame was useful
        if (!descriptors.empty() && kpts.size() > min_descriptor_count_)
        {
          trainer_.add(descriptors);
          train_count++;
          ROS_INFO_STREAM("[Haloc:] Added to trainer" << " (" << train_count << " / " << max_images_ << ").");

          // Add the frame to the sample pile
          frames_sampled_.push_back(path);
        }
        else
        {
          ROS_DEBUG("[Haloc:] Image not descriptive enough, ignoring.");
        }

        if ((!(train_count < max_images_) && max_images_ > 0))
        {
          break;
        }

        // Next directory entry
        it++;
      }
      ROS_INFO_STREAM("[Haloc:] Number of processed descriptors: "  << all_descriptors.rows);

      // Save BoW
      shutdown();

      // VLAD clustering initialization
      ROS_INFO("[Haloc:] VLAD clustering...");
      VlVectorComparisonType distance = VlDistanceL2;
      VlKMeansAlgorithm algorithm = VlKMeansLloyd;
      vl_size num_data = all_descriptors.rows;
      kmeans_ = vl_kmeans_new (VL_TYPE_FLOAT, distance);
      float *data = new float[dimension_ * num_data];

      // Re-format the data matrix
      vl_size data_idx, d;
      for(data_idx=0; data_idx<num_data; data_idx++) {
        for(d=0; d<dimension_; d++) {
          data[data_idx*dimension_+d] = all_descriptors.at<float>(data_idx, d);
        }
      }

      // kmeans settings
      vl_kmeans_set_verbosity(kmeans_, 1);
      vl_kmeans_set_max_num_iterations(kmeans_, maxiter_);
      vl_kmeans_set_max_num_comparisons(kmeans_, maxcomp_);
      vl_kmeans_set_num_repetitions(kmeans_, maxrep_);
      vl_kmeans_set_num_trees(kmeans_, ntrees_);
      vl_kmeans_set_algorithm(kmeans_, algorithm);

      // VLAD clustering
      vl_kmeans_cluster(kmeans_, data, dimension_, num_data, num_centers_);
      ROS_INFO("[Haloc:] VLAD clustering completed!");
    }

    // Openfabmap running stage
    void run()
    {
      // Load trained data
      loadCodebook();

      // Sort directory of images
      typedef std::vector<fs::path> vec;
      vec v;
      copy(
        fs::directory_iterator(running_dir_),
        fs::directory_iterator(),
        back_inserter(v)
      );
      sort(v.begin(), v.end());
      vec::const_iterator it(v.begin());

      // Read the ground truth file
      vector< vector<int> > ground_truth;
      if (!gt_file_.empty())
      {
        ifstream in(gt_file_.c_str());
        if (!in) ROS_ERROR("[Haloc:] Ground truth file does not exist.");
        for (uint x=0; x<v.size(); x++)
        {
          vector<int> row;
          for (uint y=0; y<v.size(); y++)
          {
            int num;
            in >> num;
            row.push_back(num);
          }
          ground_truth.push_back(row);
        }
        in.close();
      }

      // Initialization
      int image_id = -1;
      bool first_frame = true;
      vector<int> to_img_seq;

      // Set of correspondences image <--> VLAD matrix
      std:map<int, Mat> map_vlad;

      // Iterate over all images
      while (it!=v.end())
      {

        if (fs::is_directory(*it))
        {
          // Next directory entry and continue
          it++;
          continue;
        }

        // Read image
        string filename = it->filename().string();
        string path = running_dir_ + "/" + filename;
        Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
        image_id++;

        cout << endl << endl;
        ROS_INFO_STREAM("++++++++++++++++++++++ " << image_id << " ++++++++++++++++++++++");




        // HALOC ---------------------------------------
        lc_.setNode(img);
        vector< pair<int,float> > hash_matching;
        ros::WallTime haloc_start = ros::WallTime::now();
        lc_.getCandidates(hash_matching);
        ros::WallDuration haloc_time = ros::WallTime::now() - haloc_start;
        cout << "HALOC IMAGES [ ";
        for (int i=0; i<hash_matching.size();i++)
          cout << hash_matching[i].first << " ";
        cout << "]   \t"  << haloc_time.toSec() << " sec." << endl;
        cout << "HALOC PROB [ ";
        for (int i=0; i<hash_matching.size();i++)
          cout << round(hash_matching[i].second*100)/100 << " ";
        cout << "]" << endl;



        // OPENFABMAP ----------------------------------
        Mat bow;
        vector<KeyPoint> kpts;
        detector_->detect(img, kpts);
        bide_->compute(img, kpts, bow);

        vector<int> matched_to_img_seq;
        vector<double> matched_to_img_match;

        // Check if the frame could be described
        ros::WallTime bow_start = ros::WallTime::now();
        if (!bow.empty() && kpts.size() > min_descriptor_count_)
        {
          // IF NOT the first frame processed
          if (!first_frame)
          {
            vector<of2::IMatch> matches;

            // Find match likelyhoods for this 'bow'
            fabMap_->compare(bow, matches, !add_only_new_places_);

            // Sort matches with overloaded '<' into
            // Ascending 'match' order
            sort(matches.begin(), matches.end());

            // Add BOW
            if (add_only_new_places_)
            {
              // Check if fabMap believes this to be a new place
              if (matches.back().imgIdx == -1)
              {
                ROS_WARN("[Haloc:] Adding bow of new place...");
                fabMap_->add(bow);

                // Store the mapping to match ID
                to_img_seq.push_back(image_id);
              }
            }
            else
            {
              // Store the mapping to match ID
              to_img_seq.push_back(image_id);
            }

            // Prepare in descending match likelihood order
            int match_img_seq;
            for (vector<of2::IMatch>::reverse_iterator match_iter = matches.rbegin(); match_iter != matches.rend(); ++match_iter)
            {
              // Limit the number of matches published (by 'maxMatches_' OR 'minMatchValue_')
              if ( (matched_to_img_seq.size() == max_matches_ && max_matches_ != 0) || match_iter->match < min_match_value_)
                break;

              // Lookup IMAGE seq number from MATCH seq number
              match_img_seq = match_iter->imgIdx > -1 ? to_img_seq.at(match_iter->imgIdx) : -1;

              // Additionally if required,
              // --do NOT return matches below self matches OR new places ('-1')
              if ((match_img_seq >= image_id-self_match_window_ && disable_self_match_) || (match_img_seq == -1 && disable_unknown_match_))
                continue;

              // Add the Image seq number and its match likelihood
              matched_to_img_seq.push_back(match_img_seq);
              matched_to_img_match.push_back(match_iter->match);
            }
          }
          else
          {
            // First frame processed
            fabMap_->add(bow);
            first_frame = false;
          }
        }

        // Log
        ros::WallDuration bow_time = ros::WallTime::now() - bow_start;
        cout << "FABMAP IMAGES [ ";
        for (int i=0; i<matched_to_img_seq.size();i++)
          cout << matched_to_img_seq[i] << " ";
        cout << "]   \t"  << bow_time.toSec() << " sec." << endl;
        cout << "FABMAP PROB [ ";
        for (int i=0; i<matched_to_img_match.size();i++)
          cout << matched_to_img_match[i] << " ";
        cout << "]" << endl;



        // VLAD ----------------------------------
        kpts.clear();
        Mat descriptors;
        detector_->detect(img, kpts);
        extractor_->compute(img, kpts, descriptors);

        // Initialize and re-format input data
        vl_size num_data_to_encode = descriptors.rows;
        vl_uint32 *indexes = new vl_uint32[num_data_to_encode];
        float *distances = new float[num_data_to_encode] ;
        float *desc_to_encode = new float[num_data_to_encode * dimension_];
        for(int i=0; i<num_data_to_encode; i++)
        {
          for(int d=0; d<dimension_; d++)
            desc_to_encode[i*dimension_ + d] = descriptors.at<float>(i,d);
        }

        // Find nearest cluster centers for the data that should be encoded
        ros::WallTime vlad_start = ros::WallTime::now();
        vl_kmeans_quantize(kmeans_, indexes, distances, desc_to_encode ,num_data_to_encode);
        float *assignments = new float[num_data_to_encode * num_centers_];
        std::fill_n(assignments, num_data_to_encode * num_centers_, 0);
        for(int i=0; i<num_data_to_encode; i++)
          assignments[i * num_centers_ + indexes[i]] = 1.0;

        // Variable "enc" will store the vlad global descriptor
        float *enc = new float[dimension_ * num_centers_];
        std::fill_n(enc, dimension_ * num_centers_ , 0);
        vl_kmeans_get_centers(kmeans_);
        vl_vlad_encode(enc, VL_TYPE_FLOAT, kmeans_->centers, dimension_, num_centers_, desc_to_encode, num_data_to_encode, assignments, 0) ;
        Mat enc_mat = cv::Mat(dimension_, num_centers_, CV_32F, enc);
        transpose(enc_mat, enc_mat);

        // Store for future comparisons
        map_vlad[image_id] = enc_mat;

        // Search for loop closures
        vector< pair<int,float> > vlad_candidates;
        for (int i=0; i<(image_id - min_neighbor_-1); i++)
        {
          float msquared = 0.0;
          for(int j=0; j<enc_mat.rows; j++)
          {
            float scalar_p = 0.0;
            for(int d=0; d<enc_mat.cols; d++)
              scalar_p += enc_mat.at<float>(j,d) * map_vlad[i].at<float>(j,d);
            msquared += scalar_p;
          }
          vlad_candidates.push_back(make_pair(i, msquared));
        }

        vector<int> vlad_matchings;
        if (vlad_candidates.size() > 0)
        {
          // Sort the candidates to close loop
          std::sort(vlad_candidates.begin(), vlad_candidates.end(), EvaluationNode::sortBydistance);

          for(int i=0; i<n_candidates_; i++)
            vlad_matchings.push_back(vlad_candidates[i].first);
        }

        // Log
        ros::WallDuration vlad_time = ros::WallTime::now() - vlad_start;
        cout << "VLAD IMAGES [ ";
        for (int i=0; i<vlad_matchings.size();i++)
          cout << vlad_matchings[i] << " ";
        cout << "]   \t \t \t"  << vlad_time.toSec() << " sec." << endl;



        // GROUND TRUTH ----------------------------------
        vector<int> current_gt;
        if (ground_truth.size() > image_id)
        {
          // Get the current row
          vector<int> gt_row = ground_truth[image_id];
          cout << "GT [ ";
          for (uint i=0; i<gt_row.size(); i++)
          {
            if (gt_row[i] == 1)
            {
              current_gt.push_back(i);
              cout << i << " ";
            }
          }
          cout << "]" << endl;
        }


        // Check if openfabmap and haloc have detected a correct loop
        int haloc_idx = -1;
        int openfabmap_idx = -1;
        int vlad_idx = -1;
        bool haloc_valid = false;
        bool openfabmap_valid = false;
        bool vlad_valid = false;
        for (uint i=0; i<current_gt.size(); i++)
        {
          // Exit
          if (haloc_valid && openfabmap_valid && vlad_valid) break;

          // HALOC
          if (!haloc_valid)
          {
            for (uint j=0; j<hash_matching.size(); j++)
            {
              if (abs(hash_matching[j].first - current_gt[i]) < 10)
              {
                haloc_idx = j;
                haloc_valid = true;
                break;
              }
            }
          }

          // OPENFABMAP
          if (!openfabmap_valid)
          {
            for (uint j=0; j<matched_to_img_seq.size(); j++)
            {
              if (abs(matched_to_img_seq[j] - current_gt[i]) < 10)
              {
                openfabmap_idx = j;
                openfabmap_valid = true;
                break;
              }
            }
          }

          // VLAD
          if (!vlad_valid)
          {
            for (uint j=0; j<vlad_matchings.size(); j++)
            {
              if (abs(vlad_matchings[j] - current_gt[i]) < 10)
              {
                vlad_idx = j;
                vlad_valid = true;
                break;
              }
            }
          }
        }

        // If valid loop closure found, save the current index into a file
        if (haloc_valid || openfabmap_valid || vlad_valid)
        {
          // Open to append
          fstream f_output(output_file_.c_str(), ios::out | ios::app);
          f_output << fixed << setprecision(9) <<
                haloc_idx  << "," <<
                openfabmap_idx << "," <<
                vlad_idx << "," <<
                haloc_time.toSec() << "," <<
                bow_time.toSec() << "," <<
                vlad_time.toSec() <<  endl;
          f_output.close();
        }

        // Next directory entry
        it++;
      }
    }

  private:

    // Parameters
    ros::NodeHandle nhp_;
    int max_images_;
    int min_descriptor_count_;
    double cluster_size_;
    double lower_information_bound_;
    string vocab_path_;
    string cl_tree_path_;
    string trainbows_path_;
    string gt_file_;

    // OpenFABMap2
    of2::FabMap *fabMap_;
    of2::ChowLiuTree tree_;
    of2::BOWMSCTrainer trainer_;
    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor>  extractor_;
    Ptr<DescriptorMatcher> matcher_;
    Ptr<BOWImgDescriptorExtractor> bide_;
    Mat vocab_;
    Mat bows_;
    Mat cl_tree_;
    vector<string> frames_sampled_;

    int max_matches_;
    double min_match_value_;
    bool disable_self_match_;
    int self_match_window_;
    bool disable_unknown_match_;
    bool add_only_new_places_;
    Mat confusion_mat_;

    // VLAD
    vl_size dimension_;
    vl_size num_centers_;
    vl_size maxiter_ ;
    vl_size maxcomp_;
    vl_size maxrep_;
    vl_size ntrees_;
    VlKMeans *kmeans_;
    int min_neighbor_;
    int n_candidates_;

    // Finishes the learning stage
    void shutdown()
    {
      ROS_INFO("[Haloc:] Clustering to produce vocabulary");
      vocab_ = trainer_.cluster();
      ROS_INFO("[Haloc:] Vocabulary contains %d words, %d dims", vocab_.rows, vocab_.cols);

      ROS_INFO("[Haloc:] Setting vocabulary...");
      bide_->setVocabulary(vocab_);

      ROS_INFO("[Haloc:] Gathering BoW's...");
      findWords();

      ROS_INFO("[Haloc:] Making the Chow Liu tree...");
      tree_.add(bows_);
      cl_tree_ = tree_.make(lower_information_bound_);

      ROS_INFO("[Haloc:] Saving work completed...");
      saveCodebook();
    }

    // Find words
    void findWords()
    {
      Mat bow;
      vector<KeyPoint> kpts;
      for (uint i=0; i<frames_sampled_.size(); i++)
      {
        string path = frames_sampled_[i];
        Mat img = imread(path, CV_LOAD_IMAGE_COLOR);
        detector_->detect(img, kpts);
        bide_->compute(img, kpts, bow);
        bows_.push_back(bow);
      }
    }

    // Save everything
    void saveCodebook()
    {
      ROS_INFO("[Haloc:] Saving codebook...");
      FileStorage file;

      ROS_INFO_STREAM("[Haloc:] Saving Vocabulary to " << vocab_path_);
      file.open(vocab_path_, FileStorage::WRITE);
      file << "Vocabulary" << vocab_;
      file.release();

      ROS_INFO_STREAM("[Haloc:] Saving Chow Liu Tree to " << cl_tree_path_);
      file.open(cl_tree_path_, FileStorage::WRITE);
      file << "Tree" << cl_tree_;
      file.release();

      ROS_INFO_STREAM("[Haloc:] Saving Trained Bag of Words to " << trainbows_path_);
      file.open(trainbows_path_, FileStorage::WRITE);
      file << "Trainbows" << bows_;
      file.release();
    }

    // Load the codebook
    void loadCodebook()
    {
      ROS_INFO("[Haloc:] Loading codebook...");

      FileStorage file;
      file.open(vocab_path_, FileStorage::READ);
      file["Vocabulary"] >> vocab_;
      file.release();
      ROS_INFO("[Haloc:] Vocabulary with %d words, %d dims loaded", vocab_.rows, vocab_.cols);

      file.open(cl_tree_path_, FileStorage::READ);
      file["Tree"] >> cl_tree_;
      file.release();
      ROS_INFO("[Haloc:] Chow Liu Tree loaded");

      file.open(trainbows_path_, FileStorage::READ);
      file["Trainbows"] >> bows_;
      file.release();
      ROS_INFO("[Haloc:] Trainbows loaded");

      ROS_INFO("[Haloc:] Setting the Vocabulary...");
      bide_->setVocabulary(vocab_);

      ROS_INFO("[Haloc:] Initialising FabMap2 with Chow Liu tree...");

      // Create an instance of the FabMap2
      fabMap_ = new of2::FabMap2(cl_tree_, 0.39, 0, of2::FabMap::SAMPLED | of2::FabMap::CHOW_LIU);

      ROS_INFO("[Haloc:] Adding the trained bag of words...");
      fabMap_->addTraining(bows_);
    }

    // Sort vectors by distance
    static bool sortBydistance(const pair<int,float> p1, const pair<int,float> p2)
    {
      return (p1.second > p2.second);
    }
};

int main(int argc, char** argv)
{
  // Init ros node
  ros::init(argc, argv, "evaluation");
  EvaluationNode node;

  // Read the parameters
  node.readParameters();
  node.init();

  // BOW and VLAD training
  if (!node.training_dir_.empty())
  {
    // Sanity check
    if (!fs::exists(node.training_dir_) || !fs::is_directory(node.training_dir_))
    {
      ROS_ERROR_STREAM("[Haloc:] The image directory does not exists: " <<
      node.training_dir_);
      return 0;
    }

    // Training
    node.training();
  }

  // Normal execution
  if (!node.running_dir_.empty())
  {
    // Sanity check
    if (!fs::exists(node.running_dir_) || !fs::is_directory(node.running_dir_))
    {
      ROS_ERROR_STREAM("[Haloc:] The image directory does not exists: " <<
      node.running_dir_);
      return 0;
    }

    // Run
    node.run();
  }

  // Stop libhaloc
  node.lc_.finalize();

  ros::spin();
  return 0;
}