# Levine 带球 Frenet MPC 实车版

## 已实现

- `ball_dribble/frenet_mpc.py`：从 **Levine 的球轨迹** `paths/levine_ball_path.csv` 建立车辆后轴轨迹，进行 Frenet 五次多项式横向候选轨迹规划。结合地图墙壁和 LiDAR 障碍物排除不安全候选。在线性化运动学单轨模型上使用 **CVXPY + OSQP** 求解 MPC，状态 `[x,y,v,yaw]`，控制 `[accel,steer]`。考虑加速度、减速度、转向角、转向角变化率和 0–0.5 m/s 速度限制。球在前方装置中的横向偏移加入 MPC 代价函数。
- `ball_dribble/ball_frenet_mpc_node.py`：ROS 2 实车接口、0.25 s 前向延迟补偿、停车保护、Marker 之外的 `Path` 预测与参考轨迹。
- `ball_dribble/ball_recovery.py`：相机接触判定，横向脱离触发 STOP；确认停车后 REVERSE；找到可行轨迹后 REVERSE_STOP 再次停车；APPROACH 重新接近；相机确认接触后 PUSH。
- `ball_dribble/ball_detector_node.py`：原有 RGBD 位置输出保持不变；**新增** `/ball/camera_depth`（`std_msgs/Float32`）供接触判定使用。
- `launch/car.launch.py`：默认 Frenet MPC，Stanley 不启动。原版 Stanley 可用 `launch/stanley_car.launch.py` 单独启动。`sim.launch.py` 仍使用旧 Stanley 仿真。

## 相机距离的定义

接触条件中的 **0.29 m 是 D435i 深度图中心球面像素对应的原始光轴深度**（未加球半径），发布到 `/ball/camera_depth`。它不是 `/ball/position` 的 `x`（从**后轴**计算的坐标）。

接触确认：深度在 0.29 ± 0.035 m，`/ball/position` 的 `|y| ≤ 0.075 m`。已进入 PUSH 后横向读数 `|y| > 0.100 m` 就停车。若深度不再确认接触或相机数据超过 0.40 s，也会停车。前端装置几何位置 `ball_center_ahead=0.540 m` 仍需实测。

## 启动（在 F1TENTH 实车 Jetson）

1. 将整个 `ball_dribble` ROS 包放到 `~/f1tenth_ws/src/` 下。不要同时保留两个重名的 ROS package。
2. 安装/确认 ROS 2 Jazzy、`rclpy`、`cv_bridge`、`message_filters`、NumPy、SciPy、OpenCV、**cvxpy**、**osqp** 等依赖。在 ROS 环境里验证：

   ```bash
   python3 -c 'import numpy, scipy, cvxpy, osqp; print("MPC dependencies OK")'
   ```

   如 apt 源提供相应包，可使用 `sudo apt install python3-cvxpy python3-osqp`。务必让 `python3` 和 ROS 2 使用同一 Python 环境，避免 NumPy/OpenCV ABI 不匹配。

3. 构建：

   ```bash
   cd ~/f1tenth_ws
   colcon build --packages-select ball_dribble --symlink-install
   source install/setup.bash
   ```

   本包采用 **ament_python**，安装入口位于 `setup.py`，**不需要 `CMakeLists.txt`**。

4. 先启动 PF 定位、地图、LiDAR、VESC 和 RealSense。RealSense 需要启用彩色和深度并进行深度到彩色图像的对齐，例如：

   ```bash
   ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true align_depth.enable:=true
   ```

   确认已有 `/pf/pose/odom`、`/odom`、`/scan`、`/map`、`/camera/camera/color/image_raw` 和 `/camera/camera/aligned_depth_to_color/image_raw`。

5. **先禁用运动检查**：

   ```bash
   ros2 launch ball_dribble car.launch.py
   ros2 topic echo /ball/position --once
   ros2 topic echo /ball/camera_depth --once
   ros2 topic info /drive -v
   ```

   请确认 `/drive` **仅有一个控制器发布者**。开启前仔细验证相机坐标方向、相机俯角、球的位置单位、车辆后轴位置、路径与当前 Levine 地图一致；再在空旷区域、离地架或可靠急停监护下测试。

6. 完成校准后明确启动车辆运动：

   ```bash
   ros2 launch ball_dribble car.launch.py enable_motion:=true
   ```

   也可在节点已运行时执行 `ros2 param set /ball_frenet_mpc_node enable_motion true`。请勿同时启动旧 `stanley_car.launch.py`。

### 相机调试网页与 Foxglove 过滤

实车 launch 默认启动一个经过话题白名单限制的 Foxglove Bridge。它只转发地图、雷达、里程计、球位置、控制状态和规划路径，**不会向 Foxglove 提供 `/camera/**`、RealSense 深度/点云或 `/ball_detector_node/debug_image`**。请先停止任何另外启动的、未设置白名单的 Foxglove Bridge，避免它继续暴露全部相机话题或占用同一个 8765 端口。

检测器画面在 Jetson 本机缩放为最大 640 px、限制为 10 FPS，并以 JPEG quality 65 编码。浏览器访问：

```text
http://team7.local:8080/
```

如果 mDNS 不可用，请将 `team7.local` 换成 Jetson 的 IP 地址。网页仅供同一网络内调试，当前不包含认证或 TLS，请勿暴露到不可信网络。

端口可在启动时修改，也可按需关闭网页或内置 Foxglove Bridge：

```bash
ros2 launch ball_dribble real_levine.launch.py web_video_port:=8081
ros2 launch ball_dribble real_levine.launch.py start_web_video:=false
ros2 launch ball_dribble real_levine.launch.py start_foxglove_bridge:=false
```

只有在你自己启动了同样带 `topic_whitelist` 的 Bridge 时，才应使用最后一个选项。默认 Foxglove 地址仍是 `ws://team7.local:8765`。

查看模式和 RViz 路径：

```bash
ros2 topic echo /ball_frenet_mpc_node/mode
# RViz: nav_msgs/Path
# /ball_frenet_mpc_node/reference_path
# /ball_frenet_mpc_node/predicted_path
```

因安全条件进入 HOLD 后，请检查原因。在确认安全且处理传感器问题后，人工触发一次重置：

```bash
ros2 param set /ball_frenet_mpc_node reset_recovery true
ros2 param set /ball_frenet_mpc_node reset_recovery false
```

## 倒车限制（重要）

- 倒车速度 0.12 m/s，首次至少 0.20 m，最多 0.90 m，距离从实测后轴位移计算。
- 倒车前必须停车，等待至少 0.40 s 且车速小于 0.035 m/s。
- **默认必须有实时占据栅格地图和覆盖车辆正后方的有效雷达波束**。地图用于检查车身扫过区域；后方雷达用于检查新出现的障碍物。
- 常见 F1TENTH 前置 270° LiDAR 不能看到正后方。此时程序会进入 **HOLD**，不会盲目倒车。只有补充后方感知且验证覆盖范围之后，才能期待完全自动恢复。
- `require_rear_lidar=false` 可跳过雷达条件，仅可在封闭、有人监护的测试台架/仿真中使用。**不要在实车无人监护时关闭该条件**。
- 若 0.90 m 内找不到曲率和碰撞检查通过的恢复轨迹，车辆进入 HOLD，等待人工处理。
- 倒车结束后会进入 `REVERSE_STOP`，直到再次停稳才允许向前运动。

## 轨迹与几何

`levine_ball_path.csv` 中的 `x,y` 表示球希望经过的路径。控制器将每个点沿切线向后移动 `ball_center_ahead`（默认 0.540 m）作为车辆后轴参考。这个近似需要与真实碗形装置的位置匹配；暂按车身加装置长度 0.70 m、半宽 0.155 m 计算碰撞。

前进的 Frenet 候选轨迹默认在后轴参考线左右 0.12 m 内取 7 个终点偏移，低速下连续生成最多 1.9 m 的候选路径。恢复路径采用前向三次 Hermite 曲线及车身/障碍物/曲率检查。它不会完成多次倒车转向或复杂的非完整约束泊车，因此无法保证所有丢球情况都能恢复。

## 已做的检查与尚未验证的部分

`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q test/` 在当前离线环境通过 8 项测试（包含原有测试和新增的控制状态、地图/雷达、Levine 轨迹检查）。

当前环境缺少 CVXPY/OSQP 和完整 ROS 2，因此尚未验证实时 QP 求解速度、ROS 2 实节点启动、真实地图/雷达融合、实车倒车能力和实际持续带球效果。在实车上运行前必须完成这些验证。Stanley 旧版的仿真成功不代表新 MPC 已经通过实车验证。
