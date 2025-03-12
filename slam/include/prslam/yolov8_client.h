// OpenCV
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// ROS
#include <actionlib/client/simple_action_client.h>
#include <image_transport/image_transport.h>

// yolov8
#include <1/batchAction.h> // Note: "Action" is appended

#include <ros/ros.h>


namespace 1 {
typedef actionlib::SimpleActionClient<1::batchAction> Client;

class YOLOV8_Client_Batch {
public:
    typedef std::shared_ptr<YOLOV8_Client_Batch> Ptr;
    batchResultConstPtr mResult;

    YOLOV8_Client_Batch(const std::string& t_name, const bool t_spin)
        : ac(t_name, t_spin)
    {

        std::cout << "Waiting for action server to start." << std::endl;
        ac.waitForServer();

        factory_id_ = -1;
    }

    // void Segment(const std::vector<cv::Mat>& in_images, std::vector<cv::Mat>& out_label, std::vector<cv::Mat>& out_score, int out_object_num)
    void Segment(const std::vector<cv::Mat>& in_images, std::vector<cv::Mat>& out_label)
    {
        std::cout << "-----------------" << std::endl;
        int batch_size = in_images.size();
        if (batch_size <= 0) {

            return;
        }

        1::batchGoal goal;
        goal.id = ++factory_id_;
        for (size_t i = 0; i < batch_size; i++) {
            cv_bridge::CvImage cvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, in_images[i]);
            // smImages[i] = *(cvImage.toImageMsg());
            goal.image.push_back(*(cvImage.toImageMsg()));
        }

        std::cout << "Sending request: " << goal.id << std::endl;
        ac.sendGoal(goal);

        std::cout << "Request ID: " << goal.id << std::endl;

        //wait for the action to return
        bool finished_before_timeout = ac.waitForResult(ros::Duration(30.0));

        if (finished_before_timeout) {
            actionlib::SimpleClientGoalState state = ac.getState();
            ROS_INFO("Action finished: %s", state.toString().c_str());
        } else {
            ROS_INFO("Action did not finish before the time out.");
        }
        mResult = ac.getResult();
        // out_object_num = mResult->object_num;
        for (size_t i = 0; i < batch_size; i++) {
            cv::Mat label = cv_bridge::toCvCopy(mResult->label[i])->image;
            // cv::Mat score = cv_bridge::toCvCopy(mResult->score[i])->image;
            
            out_label.push_back(label);
            // out_score.push_back(score);
        }
    }

private:
    Client ac;
    long int factory_id_;
};
}
