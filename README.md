
# World Model  (Modern Reimplementation)

A 2025-style PyTorch reimplementation of Ha & Schmidhuber's *World Models* (2018)
on CarRacing-v2 with V (VAE) + M (MDN-RNN) + C (CMA-ES Controller).

pip install -r requirements.txt
pip uninstall -y box2d box2d-py gym box2d-py-fork
pip install gymnasium==0.29.1
pip install box2d==2.3.10
pip install pygame
verify enviorment
python -c "import gymnasium as gym; gym.make('CarRacing-v2'); print(' CarRacing OK')"




数据采集
<!-- python -m src.collect_data --episodes 10 --out data/car_racing_samples.npz -->
python -m src.collect_data --episodes 200 --out data/car_racing_samples.npz

训练 VAE 把图像压缩成 latent 向量 z_t：
<!-- python -m src.train_vae --data data/car_racing_samples.npz --epochs 30 -->
python -m src.train_vae --data data/car_racing_samples.npz --epochs 80 --batch 128

输出示例：
[VAE] ep1/30 loss=0.0123 rec=0.0109 kld=0.0138...
saved checkpoints/vae_final.pt
模型保存：checkpoints/vae_final.pt

训练世界动态模型 
𝑝(𝑧𝑡+1∣𝑧𝑡,𝑎𝑡)：
<!-- python -m src.train_rnn --data data/car_racing_samples.npz --vae_ckpt checkpoints/vae_final.pt --epochs 50 -->
python -m src.train_rnn --data data/car_racing_samples.npz --vae_ckpt checkpoints/vae_final.pt --epochs 150 --batch 64

输出示例：
[MDN-RNN] ep1/50 nll=1.2345...
saved checkpoints/mdnrnn_final.pt
模型保存：checkpoints/mdnrnn_final.pt

在学到的「世界模型」中（梦里 ）训练控制器策略：
<!-- python -m src.train_controller_cmaes --pop 32 --iters 10 --horizon 300 -->
python -m src.train_controller_cmaes --pop 64 --iters 100 --horizon 500

输出示例：
[CMA-ES] iter=10/10, best_f=-12.345
saved checkpoints/controller_cmaes.pt
这一步时间较长（取决于 population 和 horizon），是控制器进化训练。
先试小 horizon=100 以测试流程是否跑通。


让 AI 自己在脑中「播放未来」：
python -m src.world_simulator

输出示例：
Imagined 10 dream frames in latent world.



让在梦中学车的控制器回到真实世界：
<!-- python -m src.evaluate_controller --episodes 3 -->
python -m src.evaluate_controller --episodes 10

输出示例：
Episode 1: reward=890.23
Episode 2: reward=910.54
Episode 3: reward=905.77
Avg reward over 3 eps: 902.18


python -m src.finetune_controller_real   --controller_ckpt checkpoints/controller_cmaes.pt     --iters 40     --pop 12     --sigma 0.05

