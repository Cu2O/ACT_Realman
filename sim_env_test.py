import mujoco
from mujoco import viewer
import numpy as np

def visualize_mujoco_xml(xml_path):
    """
    使用 mujoco 和 mujoco.viewer 可视化 XML 文件。
    
    参数:
        xml_path (str): XML 文件的路径。
    """
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 创建 Viewer
    with viewer.launch_passive(model, data) as sim_viewer:
        # 初始化仿真
        mujoco.mj_resetData(model, data)

        # 设置初始状态（如果需要）
        # if model.nkey > 0:  # 如果定义了关键帧
            # data.qpos[:] = model.key_qpos[0]  # 使用关键帧初始化
        # 打印 mocap、box、机械臂末端坐标
        # 假设 mocap 名为 'mocap', box 名为 'box', 末端名为 'ee'，请根据实际名称修改
        mocap_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mocap_rm')
        box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rm_65/gripper_link')

        def print_positions():
            mocap_pos = data.xpos[mocap_id] if mocap_id != -1 else None
            box_pos = data.xpos[box_id] if box_id != -1 else None
            ee_pos = data.xpos[ee_id] if ee_id != -1 else None
            print(f"mocap pos: {mocap_pos}, box pos: {box_pos}, ee pos: {ee_pos}")

        print_positions()
        # 运行仿真
        print("Starting simulation...")
        while sim_viewer.is_running():
            mujoco.mj_step(model, data)  # 推进仿真一步
            sim_viewer.sync()  # 同步 Viewer
            # data.qpos[:] = model.key_qpos[0]  # 使用关键帧初始化
            print_positions()

        print("Simulation finished.")

# 测试函数调用
if __name__ == "__main__":
    xml_path = "./test/RMxml/bimanual_viperx_ee_transfer_cube_rm.xml"
    visualize_mujoco_xml(xml_path)