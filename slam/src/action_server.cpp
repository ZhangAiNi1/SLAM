#include "ros/ros.h"
#include "1/batchAction.h"
#include "1/batchFeedback.h"
#include "1/batchResult.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "actionlib/server/simple_action_server.h"
#include "thread"
#include "chrono"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <condition_variable>
#include "yolov8-seg.hpp"  // 导入实例分割模型的头文件


const std::vector<std::string> CLASS_NAMES = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},   {217, 83, 25},   {237, 177, 32},  {126, 47, 142},  {119, 172, 48},  {77, 190, 238},
    {162, 20, 47},   {76, 76, 76},    {153, 153, 153}, {255, 0, 0},     {255, 128, 0},   {191, 191, 0},
    {0, 255, 0},     {0, 0, 255},     {170, 0, 255},   {85, 85, 0},     {85, 170, 0},    {85, 255, 0},
    {170, 85, 0},    {170, 170, 0},   {170, 255, 0},   {255, 85, 0},    {255, 170, 0},   {255, 255, 0},
    {0, 85, 128},    {0, 170, 128},   {0, 255, 128},   {85, 0, 128},    {85, 85, 128},   {85, 170, 128},
    {85, 255, 128},  {170, 0, 128},   {170, 85, 128},  {170, 170, 128}, {170, 255, 128}, {255, 0, 128},
    {255, 85, 128},  {255, 170, 128}, {255, 255, 128}, {0, 85, 255},    {0, 170, 255},   {0, 255, 255},
    {85, 0, 255},    {85, 85, 255},   {85, 170, 255},  {85, 255, 255},  {170, 0, 255},   {170, 85, 255},
    {170, 170, 255}, {170, 255, 255}, {255, 0, 255},   {255, 85, 255},  {255, 170, 255}, {85, 0, 0},
    {128, 0, 0},     {170, 0, 0},     {212, 0, 0},     {255, 0, 0},     {0, 43, 0},      {0, 85, 0},
    {0, 128, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},     {0, 0, 43},      {0, 0, 85},
    {0, 0, 128},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {73, 73, 73},    {109, 109, 109}, {146, 146, 146}, {182, 182, 182}, {219, 219, 219}, {0, 114, 189},
    {80, 183, 189},  {128, 128, 0}};

const std::vector<std::vector<unsigned int>> MASK_COLORS = {
    {255, 56, 56},  {255, 157, 151}, {255, 112, 31}, {255, 178, 29}, {207, 210, 49},  {72, 249, 10}, {146, 204, 23},
    {61, 219, 134}, {26, 147, 52},   {0, 212, 187},  {44, 153, 168}, {0, 194, 255},   {52, 69, 147}, {100, 115, 255},
    {0, 24, 236},   {132, 56, 255},  {82, 0, 133},   {203, 56, 255}, {255, 149, 200}, {255, 55, 199}};

cv::Mat res;
cv::Size size = cv::Size{640, 640};
int topk = 100;
int seg_h = 160;
int seg_w = 160;
int seg_channels = 32;
float score_thres = 0.25f;
float iou_thres = 0.65f;

std::mutex cn_task_mutex;
std::mutex cn_ready_mutex;

std::condition_variable cv_task;
std::condition_variable cv_ready;

const int batch_size = 2;
bool bStartYOLOV8 = false;
cv::Mat color_image[batch_size];
std::vector<sensor_msgs::Image> masked_images;

class SemanticActionServer
{
public:
    using ActionServer = actionlib::SimpleActionServer<1::batchAction>;
    using Feedback = 1::batchFeedback;
    using Result = 1::batchResult;

    SemanticActionServer(ros::NodeHandle &nh, std::string action_name) : as_(nh, action_name, boost::bind(&SemanticActionServer::execute_cb, this, _1), false)
    {
        ROS_INFO("Initialize Action Server");
        ROS_INFO_STREAM("Action name: " << action_name);
        as_.start();
        ROS_INFO("yolov8 action server start...");
    }

    void execute_cb(const 1::batchGoalConstPtr &goal)
    {
        std::unique_lock<std::mutex> lock(cn_task_mutex);
        color_image[0] = cv_bridge::toCvCopy(goal->image[0], "bgr8")->image;
        color_image[1] = cv_bridge::toCvCopy(goal->image[1], "bgr8")->image;
        lock.unlock();

        auto timer_start = std::chrono::system_clock::now();

        ROS_INFO_STREAM("----------------------------------");
        ROS_INFO_STREAM("ID: " << goal->id);

        {
            std::unique_lock<std::mutex> lock(cn_task_mutex);
            cv_task.notify_one();  // 通知 worker 线程有新任务
        }

        {
            std::unique_lock<std::mutex> lock(cn_ready_mutex);
            cv_ready.wait(lock);  // 等待 worker 线程处理完成
            ROS_INFO("semantic result ready");
        }

        auto timer_end = std::chrono::system_clock::now();
        auto segment_time = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start).count();
        ROS_INFO_STREAM("yolov8 segment time: " << segment_time << " ms");

        result_.id = goal->id;
        result_.label = masked_images;
        feedback_.complete = true;
        as_.setSucceeded(result_);
        as_.publishFeedback(feedback_);
    }

private:
    ActionServer as_;
    Feedback feedback_;
    Result result_;
};

class YOLOV8
{
public:
    YOLOV8() : yolov8_(nullptr)
    {
        try
        {
            yolov8_ = new YOLOv8_seg("/home/plane/下载/yolov8n-seg.engine");
            if (yolov8_ == nullptr)
            {
                throw std::bad_alloc(); // Throw std::bad_alloc if allocation fails
            }
            yolov8_->make_pipe(true);
            std::cout << "=============Initialized YOLOV8==============" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to initialize YOLOV8: " << e.what() << std::endl;
        }
    }


    ~YOLOV8()
    {
        delete yolov8_;
    }

    std::vector<sensor_msgs::Image> segment(const std::vector<cv::Mat> &images)
    {
        std::vector<sensor_msgs::Image> label_msgs;
        std::vector<Object> objs_0, objs_1;

        // 调整图像输入和预处理逻辑，确保适用于实例分割模型
        yolov8_->copy_from_Mat(images[0], size);
        yolov8_->infer();  // 执行实例分割模型的推理
        yolov8_->postprocess(objs_0, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);

        yolov8_->copy_from_Mat(images[1], size);
        yolov8_->infer();  // 执行实例分割模型的推理
        yolov8_->postprocess(objs_1, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);

        // 在图像上绘制实例分割结果
        cv::Mat res_0, res_1;
        yolov8_->draw_objects(images[0], res_0, objs_0, CLASS_NAMES, COLORS, MASK_COLORS);
        yolov8_->draw_objects(images[1], res_1, objs_1, CLASS_NAMES, COLORS, MASK_COLORS);

        // 将结果保存在label_msgs中
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", res_0).toImageMsg();
        label_msgs.push_back(*msg);
        msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", res_1).toImageMsg();
        label_msgs.push_back(*msg);

        return label_msgs;
    }


private:
    YOLOv8_seg* yolov8_;


};

void worker(YOLOV8 &yolov8)
{
    bStartYOLOV8 = true;

    while (bStartYOLOV8)
    {
        std::unique_lock<std::mutex> lock(cn_task_mutex);
        cv_task.wait(lock);
        ROS_INFO("New task comming");

        masked_images = yolov8.segment({color_image[0], color_image[1]});
        {
            std::unique_lock<std::mutex> lock(cn_ready_mutex);
            cv_ready.notify_all();
        }
        // Additional logic for publishing result if needed
        // ...
    }

    ROS_INFO("Exit yolov8 thread");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolov8_server");
    ros::NodeHandle nh;

    YOLOV8 yolov8;
    std::thread worker_thread(worker, std::ref(yolov8));

    SemanticActionServer action_server(nh, "/yolov8_action_server");

    ROS_INFO("Setting up yolov8 Action Server...");

    
    ros::spin();
    bStartYOLOV8 = false;
    
    worker_thread.join();

    return 0;
}

