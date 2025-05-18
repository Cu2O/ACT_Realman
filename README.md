# ACT_Realman
Aloha ACT execute on Realman65，Aloha ACT在Realman65机械臂上复现

# 数据集收集
python3 record_sim_episodes_rm.py --task_name sim_grasp_cube_ur --dataset_dir data/sim_insertion --num_episodes 50 --onscreen_render

# 数据训练
python3 imitate_episodes_rm.py --task_name sim_grasp_cube_ur --ckpt_dir ./model --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 128 --batch_size 4 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
