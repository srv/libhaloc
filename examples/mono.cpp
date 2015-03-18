#include <ros/ros.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "libhaloc/lc.h"

using namespace std;
namespace fs=boost::filesystem;

/**
 * Executes libhaloc over a set of images
 */
class Mono
{
  public:

    /** \brief Mono class constructor
     */
    Mono(ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_private_(nhp)
    {
      // Read the parameters
      readParams();
    }

    /** \brief Processes the image dataset
     */
    void processData()
    {
      // Initialize output files
      string log_file_str = output_path_ + "log" + tmp_id_ + ".txt";
      string out_file_str = output_path_ + "out" + tmp_id_ + ".txt";

      // Delete files if exist
      fs::wpath log_file(log_file_str);
      if(fs::exists(log_file)) fs::remove(log_file);
      fs::wpath out_file(out_file_str);
      if(fs::exists(out_file)) fs::remove(out_file);

      // Sort directory of images
      typedef vector<fs::path> vec;
      vec v;
      copy(
            fs::directory_iterator(img_dir_),
            fs::directory_iterator(),
            back_inserter(v)
          );

      sort(v.begin(), v.end());
      vec::const_iterator it(v.begin());

      // Read the ground truth file
      int total_lc = 0;
      vector< vector<int> > ground_truth;
      ifstream in(gt_file_.c_str());
      if (!in)
      {
        ROS_WARN("[HashMatching: ] Ground truth file does not exist.");
      }
      else
      {
        for (int x=0; x<(int)v.size(); x++)
        {
          vector<int> row;
          for (int y=0; y<(int)v.size(); y++)
          {
            int num;
            in >> num;
            row.push_back(num);
          }
          ground_truth.push_back(row);
          int sum_of_elems = accumulate(row.begin(),row.end(),0);
          if (sum_of_elems > 0)
            total_lc++;
        }
      }
      in.close();

      // Init Haloc
      haloc::LoopClosure::Params lc_params;
      lc_params.work_dir = output_path_;
      lc_params.desc_type = desc_type_;
      lc_params.desc_matching_type = desc_matching_type_;
      lc_params.desc_thresh_ratio = desc_thresh_ratio_;
      lc_params.epipolar_thresh = epipolar_thresh_;
      lc_params.min_neighbor = min_neighbor_;
      lc_params.n_candidates = n_candidates_;
      lc_params.min_matches = min_matches_;
      lc_params.min_inliers = min_inliers_;
      lc_.setParams(lc_params);
      lc_.init();

      // Count the overall loop time
      ros::WallTime overall_time_start = ros::WallTime::now();

      // Iterate over all images
      int img_i = 0;
      bool first = true;
      int found_lc = 0;
      int true_positives = 0;
      int false_positives = 0;
      vector<string> image_filenames;
      while (it!=v.end())
      {
        // Check if the directory entry is an directory.
        if (!fs::is_directory(*it))
        {
          // Get image
          string cur_filename = it->filename().string();
          Mat img = imread(img_dir_+"/"+cur_filename, CV_LOAD_IMAGE_COLOR);
          image_filenames.push_back(cur_filename);

          // Set the new image
          lc_.setNode(img);

          // Get the loop closure (if any)
          int img_lc = -1;
          bool valid = lc_.getLoopClosure(img_lc);

          // Check ground truth
          int tp = 0;
          int fa = 0;
          if (valid)
          {
            found_lc++;
            int gt_valid = 0;
            if (ground_truth.size() > 0)
            {
              for (int i=0; i<2*gt_tolerance_+1; i++)
              {
                int img_j = img_lc - gt_tolerance_ + i;
                if (img_j<0) img_j = 0;
                gt_valid += ground_truth[img_i][img_j];
              }
            }

            if(gt_valid >= 1)
            {
              true_positives++;
              tp = 1;
            }
            else
            {
              false_positives++;
              fa = 1;
            }
          }

          img_i++;

          string lc_filename = "";
          if (img_lc >= 0)
          {
            lc_filename = image_filenames[img_lc];
          }

          // Log
          ROS_INFO_STREAM( cur_filename << " cl with " << img_lc << ": " << valid << ": " << tp << "|" << fa);
          fstream f_log(log_file_str.c_str(), ios::out | ios::app);
          f_log << cur_filename << "," <<
                   lc_filename << "," <<
                   img_lc << "," <<
                   valid << "," <<
                   tp << "," <<
                   fa << endl;
          f_log.close();
        }

        // Next directory entry
        it++;
      }

      // Finalize the loop closure class
      lc_.finalize();

      // Stop time
      ros::WallDuration overall_time = ros::WallTime::now() - overall_time_start;

      // Compute precision and recall
      int false_negatives = total_lc - found_lc;
      double precision = round( 100 * true_positives / (true_positives + false_positives) );
      double recall = round( 100 * true_positives / (true_positives + false_negatives) );

      // Print the results
      ROS_INFO_STREAM("TOTAL #LC: " << total_lc);
      ROS_INFO_STREAM("FOUND #LC: " << found_lc);
      ROS_INFO_STREAM("#TP: " << true_positives);
      ROS_INFO_STREAM("#FP: " << false_positives);
      ROS_INFO_STREAM("PRECISION: " << precision << "%");
      ROS_INFO_STREAM("RECALL: " << recall << "%");
      ROS_INFO_STREAM("TOTAL EXECUTION TIME: " << overall_time.toSec() << " sec.");

      // Append results to file
      fstream f_out(out_file_str.c_str(), fstream::in | fstream::out | fstream::app);
      f_out <<  desc_type_ << "," <<
                desc_matching_type_ << "," <<
                desc_thresh_ratio_ << "," <<
                min_neighbor_ << "," <<
                n_candidates_ << "," <<
                min_matches_ << "," <<
                min_inliers_ << "," <<
                epipolar_thresh_ << "," <<
                gt_tolerance_ << "," <<
                total_lc << "," <<
                found_lc << "," <<
                true_positives << "," <<
                false_positives << "," <<
                precision << "," <<
                recall << "," <<
                overall_time.toSec() << endl;
      f_out.close();
    }

  protected:

    // Node handlers
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

  private:

    // Properties
    string tmp_id_, img_dir_, desc_type_, desc_matching_type_, output_path_, gt_file_;
    double desc_thresh_ratio_, epipolar_thresh_;
    int min_neighbor_, n_candidates_, min_matches_, min_inliers_, gt_tolerance_;
    haloc::LoopClosure lc_;

    /** \brief Read the parameters from the ros parameter server
     */
    void readParams()
    {
      nh_private_.param("tmp_id", tmp_id_, std::string(""));
      nh_private_.param("output_path", output_path_, std::string(""));
      nh_private_.param("img_dir", img_dir_, std::string(""));
      nh_private_.param("gt_file", gt_file_, std::string(""));
      nh_private_.param("desc_type", desc_type_, std::string("SIFT"));
      nh_private_.param("desc_matching_type", desc_matching_type_, std::string("CROSSCHECK"));
      nh_private_.getParam("desc_thresh_ratio", desc_thresh_ratio_);
      nh_private_.getParam("min_neighbor", min_neighbor_);
      nh_private_.getParam("n_candidates", n_candidates_);
      nh_private_.getParam("min_matches", min_matches_);
      nh_private_.getParam("min_inliers", min_inliers_);
      nh_private_.getParam("epipolar_thresh", epipolar_thresh_);
      nh_private_.getParam("gt_tolerance", gt_tolerance_);

      // Log
      cout << "  tmp_id            = " << tmp_id_ << endl;
      cout << "  output_path       = " << output_path_ << endl;
      cout << "  img_dir           = " << img_dir_ << endl;
      cout << "  desc_type         = " << desc_type_ << endl;
      cout << "  desc_thresh_ratio = " << desc_thresh_ratio_ << endl;
      cout << "  min_neighbor      = " << min_neighbor_ << endl;
      cout << "  n_candidates      = " << n_candidates_ << endl;
      cout << "  min_matches       = " << min_matches_ << endl;
      cout << "  min_inliers       = " << min_inliers_ << endl;
      cout << "  epipolar_thresh   = " << epipolar_thresh_ << endl;
      cout << "  gt_tolerance      = " << gt_tolerance_ << endl;

      // Files path sanity check
      if (output_path_[output_path_.length()-1] != '/')
        output_path_ += "/";

      // Sanity checks
      if (!fs::exists(img_dir_) || !fs::is_directory(img_dir_))
      {
        ROS_ERROR_STREAM("[HashMatching:] The image directory does not exists: " <<
                         img_dir_);
      }
    }
};

// Main entry point
int main(int argc, char **argv)
{
  ros::init(argc,argv,"example_mono");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  // Init node
  Mono mono(nh,nh_private);

  // Process the data
  mono.processData();

  // Subscription is handled at start and stop service callbacks.
  //ros::spin();

  return 0;
}

