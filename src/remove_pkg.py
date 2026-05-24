import shutil
try:
    shutil.rmtree('/home/adarsh/ros2_ws/src/av_fusion_msgs')
    print("Successfully removed av_fusion_msgs")
except Exception as e:
    print(f"Error: {e}")
