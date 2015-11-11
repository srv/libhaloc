#include <ros/ros.h>
#include <ros/package.h>
#include <numeric>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "libhaloc/simple_hash.h"
#include "libhaloc/utils.h"

using namespace std;
using namespace boost;

namespace fs=filesystem;

class Haloc
{
  haloc::SimpleHash hash_;
  vector< pair<int, vector<float> > > hash_table_;
  string execution_dir_;

  public:
  Haloc(string input_img_dir)
  {
    string img_dir = input_img_dir;

    // Init
    execution_dir_ = ros::package::getPath("libhaloc") + "/" + "haloc";
    if (fs::is_directory(execution_dir_))
      fs::remove_all(execution_dir_);
    fs::path dir1(execution_dir_);
    if (!fs::create_directory(dir1))
      ROS_ERROR("[Localization:] ERROR -> Impossible to create the loop_closing directory.");

    // Sort directory of images
    typedef std::vector<fs::path> vec;
    vec v;
    copy(
        fs::directory_iterator(img_dir),
        fs::directory_iterator(),
        back_inserter(v)
        );

    sort(v.begin(), v.end());
    vec::const_iterator it(v.begin());

    ostringstream output_csv;

    int id = 0;
    int lc = 0;

    // Iterate over all images
    while (it!=v.end())
    {
      // Check if the directory entry is an directory.
      if (!fs::is_directory(*it))
      {
        // Get filename
        string filename = it->filename().string();
        int lastindex = filename.find_last_of(".");
        string rawname = filename.substr(0, lastindex);

        // Read image
        string path = img_dir + "/" + filename;
        cv::Mat img_cur = cv::imread(path, CV_LOAD_IMAGE_COLOR);

        // Extract kp
        vector<cv::KeyPoint> tmp_kp;
        cv::ORB orb(1500, 1.2, 8, 10, 0, 2, 0, 10);
        orb(img_cur, cv::noArray(), tmp_kp, cv::noArray(), false);

        // Bucket kp
        vector<cv::KeyPoint> q_kp = hash_.bucketFeatures(tmp_kp);

        // Extract descriptors
        cv::Mat q_desc;
        cv::Ptr<cv::DescriptorExtractor> cv_extractor;
        cv::initModule_nonfree();
        cv_extractor = cv::DescriptorExtractor::create("SIFT");
        cv_extractor->compute(img_cur, q_kp, q_desc);

        // Initialize hash
        if (!hash_.isInitialized())
          hash_.init(q_desc);

        // Save hash to table
        vector<float> q_hash = hash_.getHash(q_desc);
        hash_table_.push_back(make_pair(id, q_hash));

        // Store
        cv::FileStorage fs(execution_dir_+"/"+lexical_cast<string>(id)+".yml", cv::FileStorage::WRITE);
        write(fs, "desc", q_desc);
        fs.release();

        // Get the candidate to close loop
        vector< pair<int,float> > candidates;
        getCandidates(id, candidates);

        if (candidates.size() > 0)
        {
          // Read the descriptors of the first candidate
          cv::Mat c_desc = readDesc(candidates[0].first);

          // Read the descriptors of the candidate
          Mat match_mask;
          vector<DMatch> matches;
          haloc::Utils::crossCheckThresholdMatching(q_desc,
              c_desc, 0.8, match_mask, matches);

          if (id == 251)
          {
            ROS_INFO_STREAM("KKKKK: " << candidates[0].first << " - " << matches.size());

            // Save the query descriptors
            ostringstream query_desc_csv;
            for (int n=0; n<q_desc.rows; n++)
            {
              for (int m=0; m<q_desc.cols; m++)
                query_desc_csv << q_desc.at<float>(n, m) << ",";
              query_desc_csv.seekp(query_desc_csv.str().length()-1);
              query_desc_csv << endl;
            }
            // Save to file
            string out_q_desc_file = ros::package::getPath("libhaloc") + "/q_desc.txt";
            fstream f_q_desc_file(out_q_desc_file.c_str(), ios::out | ios::trunc);
            f_q_desc_file << query_desc_csv.str();
            f_q_desc_file.close();

            // Save the candidate descriptors
            ostringstream cand_desc_csv;
            for (int n=0; n<c_desc.rows; n++)
            {
              for (int m=0; m<c_desc.cols; m++)
                cand_desc_csv << c_desc.at<float>(n, m) << ",";
              cand_desc_csv.seekp(cand_desc_csv.str().length()-1);
              cand_desc_csv << endl;
            }
            // Save to file
            string out_c1_desc_file = ros::package::getPath("libhaloc") + "/108_desc.txt";
            fstream f_c1_desc_file(out_c1_desc_file.c_str(), ios::out | ios::trunc);
            f_c1_desc_file << cand_desc_csv.str();
            f_c1_desc_file.close();

            // Save the candidate descriptors
            c_desc = readDesc(107);
            ostringstream cand1_desc_csv;
            for (int n=0; n<c_desc.rows; n++)
            {
              for (int m=0; m<c_desc.cols; m++)
                cand1_desc_csv << c_desc.at<float>(n, m) << ",";
              cand1_desc_csv.seekp(cand1_desc_csv.str().length()-1);
              cand1_desc_csv << endl;
            }
            // Save to file
            string out_c2_desc_file = ros::package::getPath("libhaloc") + "/107_desc.txt";
            fstream f_c2_desc_file(out_c2_desc_file.c_str(), ios::out | ios::trunc);
            f_c2_desc_file << cand1_desc_csv.str();
            f_c2_desc_file.close();
            return;
          }

          // Is it a possible loop?
          if (matches.size() > 40)
          {

            // Log
            ROS_INFO_STREAM(id << " to " << candidates[0].first << ": " << matches.size());

            /*
            // Save the query descriptors
            ostringstream query_desc_csv;
            for (int n=0; n<q_desc.rows; n++)
            {
              for (int m=0; m<q_desc.cols; m++)
                query_desc_csv << q_desc.at<float>(n, m) << ",";
              query_desc_csv.seekp(query_desc_csv.str().length()-1);
              query_desc_csv << endl;
            }
            // Save to file
            string out_q_desc_file = ros::package::getPath("libhaloc") + "/q_desc.txt";
            fstream f_q_desc_file(out_q_desc_file.c_str(), ios::out | ios::trunc);
            f_q_desc_file << query_desc_csv.str();
            f_q_desc_file.close();

            // Save the candidate descriptors
            ostringstream cand_desc_csv;
            for (int n=0; n<c_desc.rows; n++)
            {
              for (int m=0; m<c_desc.cols; m++)
                cand_desc_csv << c_desc.at<float>(n, m) << ",";
              cand_desc_csv.seekp(cand_desc_csv.str().length()-1);
              cand_desc_csv << endl;
            }
            // Save to file
            string out_c1_desc_file = ros::package::getPath("libhaloc") + "/lc_desc.txt";
            fstream f_c1_desc_file(out_c1_desc_file.c_str(), ios::out | ios::trunc);
            f_c1_desc_file << cand_desc_csv.str();
            f_c1_desc_file.close();

            // Save matches indices
            ostringstream matches_q2lc;
            for (uint i=0; i<matches.size(); i++)
              matches_q2lc << matches[i].queryIdx << "," << matches[i].trainIdx << endl;
            // Save to file
            string out_q2lc_file = ros::package::getPath("libhaloc") + "/matches_q2lc.txt";
            fstream f_q2lc_file(out_q2lc_file.c_str(), ios::out | ios::trunc);
            f_q2lc_file << matches_q2lc.str();
            f_q2lc_file.close();

            // Save hashes
            ostringstream query_hash;
            for (uint i=0; i<q_hash.size(); i++)
              query_hash << q_hash[i] << endl;
            // Save to file
            string out_qh_file = ros::package::getPath("libhaloc") + "/query_hash.txt";
            fstream f_qh_file(out_qh_file.c_str(), ios::out | ios::trunc);
            f_qh_file << query_hash.str();
            f_qh_file.close();

            ostringstream lc_hash;
            vector<float> c_hash = hash_table_[candidates[0].first].second;
            for (uint i=0; i<c_hash.size(); i++)
              lc_hash << c_hash[i] << endl;
            // Save to file
            string out_lch_file = ros::package::getPath("libhaloc") + "/lc_hash.txt";
            fstream f_lch_file(out_lch_file.c_str(), ios::out | ios::trunc);
            f_lch_file << lc_hash.str();
            f_lch_file.close();
            */

            output_csv << matches.size() << "," << candidates[0].second << endl;
            for (uint i=1; i<candidates.size(); i++)
            {
              c_desc = readDesc(candidates[i].first);
              haloc::Utils::crossCheckThresholdMatching(q_desc,
                  c_desc, 0.8, match_mask, matches);

              /*
              if (matches.size() < 5)
              {
                // Save the descriptor matrix
                ostringstream nolc_desc_csv;
                for (int n=0; n<c_desc.rows; n++)
                {
                  for (int m=0; m<c_desc.cols; m++)
                    nolc_desc_csv << c_desc.at<float>(n, m) << ",";
                  nolc_desc_csv.seekp(nolc_desc_csv.str().length()-1);
                  nolc_desc_csv << endl;
                }
                // Save to file
                string out_c2_desc_file = ros::package::getPath("libhaloc") + "/no_lc_desc.txt";
                fstream f_c2_desc_file(out_c2_desc_file.c_str(), ios::out | ios::trunc);
                f_c2_desc_file << nolc_desc_csv.str();
                f_c2_desc_file.close();

                // Save matches indices
                ostringstream matches_q2nolc;
                for (uint j=0; i<matches.size(); j++)
                  matches_q2nolc << matches[j].queryIdx << "," << matches[j].trainIdx << endl;
                // Save to file
                string out_q2nolc_file = ros::package::getPath("libhaloc") + "/matches_q2nolc.txt";
                fstream f_q2nolc_file(out_q2nolc_file.c_str(), ios::out | ios::trunc);
                f_q2nolc_file << matches_q2nolc.str();
                f_q2nolc_file.close();

                ostringstream nolc_hash;
                c_hash = hash_table_[candidates[i].first].second;
                for (uint j=0; j<c_hash.size(); j++)
                  nolc_hash << c_hash[j] << endl;
                // Save to file
                string out_nolch_file = ros::package::getPath("libhaloc") + "/no_lc_hash.txt";
                fstream f_nolch_file(out_nolch_file.c_str(), ios::out | ios::trunc);
                f_nolch_file << nolc_hash.str();
                f_nolch_file.close();

                ROS_INFO_STREAM("NO LC: " << candidates[i].first);

                return;
              }
              */

              output_csv << matches.size() << "," << candidates[i].second << endl;
            }
            // Increase the lc counter
            lc++;
          }
        }

        id++;
      }

      // Next directory entry
      it++;
    }

    // Save
    string out_file = execution_dir_ + "/output.txt";
    fstream f_out(out_file.c_str(), ios::out | ios::trunc);
    f_out << output_csv.str();
    f_out.close();

    ROS_INFO_STREAM("END. Found " << lc << " lc.");
    ros::shutdown();

  }

  Mat readDesc(int id)
  {
    // Get the image keypoints and descriptors
    cv::FileStorage fs;
    fs.open(execution_dir_+"/"+lexical_cast<string>(id)+".yml", cv::FileStorage::READ);
    if (!fs.isOpened())
      ROS_ERROR("[Haloc:] ERROR -> Failed to open the image descriptors.");
    cv::Mat desc;
    fs["desc"] >> desc;
    fs.release();
    return desc;
  }

  void getCandidates(int id, vector< pair<int,float> >& candidates)
  {
    // Init
    candidates.clear();

    // Check if enough neighbors
    if ((int)hash_table_.size() <= 10) return;

    // Query hash
    vector<float> hash_q = hash_table_[id].second;

    // Loop over all the hashes stored
    vector< pair<int,float> > all_matchings;
    for (uint i=0; i<hash_table_.size(); i++)
    {
      // Discard window
      if (hash_table_[i].first > id-10 && hash_table_[i].first < id+10) continue;

      // Do not compute the hash matching with itself
      if (hash_table_[i].first == id) continue;

      // Hash matching
      vector<float> hash_t = hash_table_[i].second;
      float m = hash_.match(hash_q, hash_t);
      all_matchings.push_back(make_pair(hash_table_[i].first, m));
    }

    // Sort the hash matchings
    sort(all_matchings.begin(), all_matchings.end(), haloc::Utils::sortByMatching);

    // Retrieve the best n matches
    uint max_size = 40;
    if (max_size > all_matchings.size()) max_size = all_matchings.size();
    for (uint i=0; i<max_size; i++)
      candidates.push_back(all_matchings[i]);

    // Normalize hash matching from 0 to 1
    float max = 0.0;
    float min = 99999999999.9;
    for (uint i=0; i<candidates.size(); i++)
    {
      if (candidates[i].second > max)
        max = candidates[i].second;
      if (candidates[i].second < min)
        min = candidates[i].second;
    }
    for (uint i=0; i<candidates.size(); i++)
      candidates[i].second = (candidates[i].second - min) / (max - min);

  }
};

int main(int argc, char** argv)
{
  // Parse arguments
  if (argc < 2) {
    // Inform the user of how to use the program
    std::cout << "Usage is: rosrun libhaloc evaluation <image directory>\n";
    std::cin.get();
    exit(0);
  }
  string work_dir = argv[1];

  ros::init(argc, argv, "libhaloc");
  Haloc node(work_dir);
  ros::spin();
  return 0;
}

