# ACT_Realman
Aloha ACT execute on Realman65，Aloha ACT在Realman65机械臂上复现

# 数据集收集
``` python3 record_sim_episodes_rm.py --task_name sim_grasp_cube_ur --dataset_dir data/sim_grasp_cube_ur --num_episodes 50 --onscreen_render``` 

# 数据训练
``` python3 imitate_episodes_rm.py --task_name sim_grasp_cube_ur --ckpt_dir ./model --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 128 --batch_size 4 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0``` 

# 修改
把 imitate_episodes_rm.py detr/main.py 中所有.cuda()替换为了.to('cpu')
在gpu训练请换为cuda。

请将RMxml文件夹放入test文件夹下。

因为控制对象由act的ALOHA系统换为了单机械臂，动作维度由14换为6，因此更新了detr/models下的文件，请注意替换。

笔者业余复现，请注意复现时原文件的保存。

# 评估
``` python3 imitate_episodes_rm.py --task_name sim_grasp_cube_ur --ckpt_dir ./model --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 128 --batch_size 4 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0  --eval  --onscreen_render``` 

# 视频

https://github.com/user-attachments/assets/4be735e8-7dd8-4fb1-9193-ef3e92e0d595




## 文件
constants.py 增加对sim_grasp_cube_ur任务的配置

ee_sim_env_rm.py 通过mocap控制机械臂末端的仿真环境

sim_env_rm.py 通过控制机械臂角度的仿真环境

record_sim_episodes_rm.py 数据集收集

imitate_episodes_rm.py act训练与评价

scripted_policy_rm.py 脚本操控机械臂收集数据

test_sim_env.py sim_env_test.py机械臂角度控制仿真环境测试

detr_vae.py 更新机械臂动作维度
