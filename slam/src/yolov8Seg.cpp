#include "ros/ros.h"
#include "sensor_msgs/CompressedImage.h"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "chrono"
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

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr& msg)
{
    try
    {
        // 将压缩图像消息解压缩成OpenCV图像
        cv::Mat image = cv::imdecode(cv::Mat(msg->data), 1);

        // 在这里添加实例分割模型的推理和后处理逻辑
        auto yolov8 = new YOLOv8_seg("/home/Downloads/yolov8n-seg.engine");
        yolov8->make_pipe(true);

        cv::Mat res;
        cv::Size size = cv::Size{640, 640};
        int topk = 100;
        int seg_h = 160;
        int seg_w = 160;
        int seg_channels = 32;
        float score_thres = 0.25f;
        float iou_thres = 0.65f;

        std::vector<Object> objs;

        // 调整图像输入和预处理逻辑，确保适用于实例分割模型
        yolov8->copy_from_Mat(image, size);

        auto start = std::chrono::system_clock::now();
        yolov8->infer();  // 执行实例分割模型的推理
        auto end = std::chrono::system_clock::now();

        // 后处理实例分割结果
        yolov8->postprocess(objs, score_thres, iou_thres, topk, seg_channels, seg_h, seg_w);

        // 在图像上绘制实例分割结果
        yolov8->draw_objects(image, res, objs, CLASS_NAMES, COLORS, MASK_COLORS);

        auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
        printf("cost %2.4lf ms\n", tc);

        // 显示图像
        cv::imshow("Compressed Image Viewer", res);
        cv::waitKey(100);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "image_listener");
    ros::NodeHandle nh;

    // 创建一个ROS订阅者，订阅压缩图像消息
    ros::Subscriber sub = nh.subscribe("/camera/rgb/image_raw/compressed", 1, imageCallback);

    // 初始化OpenCV窗口
    cv::namedWindow("Compressed Image Viewer");

    // 循环处理ROS事件
    ros::spin();

    return 0;
}
