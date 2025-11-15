# src/simple_carracing_env.py

import numpy as np

class SimpleCarRacingEnv:
    """
    一个极简版“赛车环境”，不依赖 gym/Box2D，只提供：
    - obs: (64, 64, 3) numpy.float32, 值在 [0,1]
    - action: (3,) ndarray, [steer, gas, brake] ∈ [-1,1]
    - 接口: reset(), step(), close()
    仅用于让 VAE + Controller 的评估代码跑通。
    """

    def __init__(self, max_steps: int = 500):
        self.max_steps = max_steps
        self.step_count = 0
        self.position = 0.0   # 沿赛道前进的“距离”
        self.lane_offset = 0.0  # 车偏离车道中心的程度（-1,1）
        self.done = False

    def reset(self):
        self.step_count = 0
        self.position = 0.0
        self.lane_offset = 0.0
        self.done = False
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        action: array-like, shape (3,)
        [steer, gas, brake] in [-1, 1]
        """
        if self.done:
            # 兼容 gym 风格，reset 以后才可以继续 step
            return self._get_obs(), 0.0, True, False, {}

        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,):
            raise ValueError(f"Action must be shape (3,), got {action.shape}")

        steer, gas, brake = action
        gas = np.clip(gas, -1.0, 1.0)
        brake = np.clip(brake, -1.0, 1.0)
        steer = np.clip(steer, -1.0, 1.0)

        # 简单速度模型：速度 ~ gas - max(brake, 0)
        speed = np.clip(gas - max(brake, 0.0), 0.0, 1.0)

        # 更新状态
        self.position += float(speed)
        self.lane_offset += float(0.1 * steer)  # 转向改变车道偏移
        self.lane_offset = float(np.clip(self.lane_offset, -2.0, 2.0))

        self.step_count += 1

        # 奖励：往前走 + 不偏离车道
        lane_penalty = abs(self.lane_offset)    # 偏离越大，惩罚越大
        reward = float(speed * 1.0 - 0.1 * lane_penalty)

        # 终止条件：步数用完，或者偏离太大
        terminated = False
        truncated = False

        if self.step_count >= self.max_steps:
            truncated = True
        if abs(self.lane_offset) > 1.5:
            terminated = True

        self.done = terminated or truncated

        obs = self._get_obs()
        info = {"position": self.position, "lane_offset": self.lane_offset}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        构造一个 64x64x3 的“赛道图像”：
        - 黑色背景
        - 中间一条灰色车道
        - 底部一个红色小块表示车辆
        """
        img = np.zeros((64, 64, 3), dtype=np.float32)

        # 车道区域
        lane_left = 20
        lane_right = 44
        img[:, lane_left:lane_right, :] = 0.4  # 灰色车道

        # 根据 lane_offset 把车在底部左右移动一点
        car_col_center = 32 + int(self.lane_offset * 10)
        car_col_center = int(np.clip(car_col_center, 4, 60))
        car_row_top = 50
        car_row_bottom = 60
        car_col_left = car_col_center - 2
        car_col_right = car_col_center + 2

        img[car_row_top:car_row_bottom, car_col_left:car_col_right, 0] = 1.0  # 红色小方块

        return img

    def close(self):
        pass
