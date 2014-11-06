#include "ros/ros.h"
#include "libhaloc/RemoteExecution.h"
#include <sstream>

using namespace std;

bool start_node(libhaloc::RemoteExecution::Request  &req,
         		libhaloc::RemoteExecution::Response &res)
{
	try
	{
		// Launch the node
		string cmd = 	"roslaunch libhaloc remote.launch tmp_id:=" +
							req.tmp_id + " img_dir:=" +
							req.img_dir + " gt_file:=" +
							req.gt_file + " desc_type:=" +
							req.desc_type + " desc_matching_type:=" +
							req.desc_matching_type + " desc_thresh_ratio:=" +
							req.desc_thresh_ratio + " num_proj:=" +
							req.num_proj + " min_neighbor:=" +
							req.min_neighbor + " n_candidates:=" +
							req.n_candidates + " min_matches:=" +
							req.min_matches + " min_inliers:=" +
							req.min_inliers + " gt_tolerance:=" +
							req.gt_tolerance + " &";
		system(cmd.c_str());

		// Output
		res.msg = "Node started!";
	}
	catch (exception& e)
	{
		// Output
		res.msg = "Error launching the node!";
	}

	return true;
}

bool stop_node(libhaloc::RemoteExecution::Request  &req,
         	   libhaloc::RemoteExecution::Response &res)
{
	try
	{
		// Launch the node
		string cmd = "rosnode kill remote_" + req.tmp_id;
		system(cmd.c_str());

		// Output
		res.msg = "Node stopped!";
	}
	catch (exception& e)
	{
		// Output
		res.msg = "Error killing the node!";
	}

	return true;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "libhaloc");
	ros::NodeHandle n("~");

	ros::ServiceServer start_service = n.advertiseService("start_remote_exec", start_node);
	ros::ServiceServer stop_service  = n.advertiseService("stop_remote_exec", stop_node);
	ros::spin();
	return 0;
}