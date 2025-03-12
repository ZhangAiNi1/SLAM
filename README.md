# Enhanced Semantic SLAM for Dynamic Indoor Scenes A Probabilistic Approach
Simultaneous Localization and Mapping (SLAM) technology is pivotal for autonomous exploration in unknown environments. Traditional SLAM systems predominantly rely on static scene assumptions, limiting their applicabil-ity in dynamic real-world scenarios. To address this, we propose an enhanced semantic SLAM method tailored for dynamic indoor environments. Built upon the ORB-SLAM3 framework, our system integrates real-time semantic segmentation using YOLOv8 and employs dynamic probability propagation to precisely locate static feature points within potentially dynamic regions. By incorporating depth image geometric information, we optimize mask boundaries, effectively eliminating noise residues in point cloud maps. Through multi-level feature screening and mask refinement, our method achieves high-precision global point cloud maps. Here, we show that our approach outperforms classical SLAM and other dynamic SLAM algorithms on the TUM dataset, demonstrating substantial improvements in both Absolute Trajectory Error (ATE) and Relative Pose Error (RPE), with mean errors reduced by up to 97.77% and 97.83%, respectively. Our work contributes to advancing the robustness and accuracy of SLAM in complex, dynamic environments.
# Usage

catkin_make -DCATKIN_WHITELIST_PACKAGES="slam_track"

conda activate yolov8

roslaunch astra_camera astra.launch

rosrun slam yolov8_action_server

roslaunch slam_track room_RGB-D.launch

roslaunch robot_driver robot_run.launch

roslaunch robot_cl keybord.launch

evo_ape tum CameraTrajectory.txt CameraTrajectory.txt -a -p -s

# Note
The code is mainly based on ORB-SLAM3(https://github.com/UZ-SLAMLab/ORB_SLAM3) .
The test dataset can be download at TUM : https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download; 
