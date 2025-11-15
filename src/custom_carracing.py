import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces


class RealisticCarRacing(gym.Env):
    """
    A Box2D-free, Windows-friendly CarRacing-like environment.

    - Top-down view
    - Procedurally generated closed track
    - Simple but smoother car dynamics than the previous toy version
    - Reward:
        * +0.05 for being alive
        * +0.1 * progress along track
        * -1.0 if far off-track
        * Episode ends if too far from track or max_steps exceeded
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode="rgb_array", obs_size=96):
        super().__init__()
        self.render_mode = render_mode
        self.obs_size = obs_size

        # Observation: RGB image
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(obs_size, obs_size, 3), dtype=np.uint8
        )

        # Action: [steer, gas, brake]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # World scale
        self.track_radius = 40.0
        self.track_width = 8.0
        self.off_track_threshold = self.track_width * 2.5

        self.max_speed = 0.8
        self.accel_power = 0.04
        self.brake_power = 0.06
        self.friction = 0.98

        self.max_steps = 1500

        self.track = None
        self.track_length = None
        self.reset_internal()
        self.generate_track()

    # ------------- core helpers -------------

    def reset_internal(self):
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.speed = 0.0
        self.heading = 0.0  # radians
        self.step_count = 0
        self.done = False

        self.closest_idx = 0  # nearest track point index
        self.progress = 0.0   # accumulated progress along track

    def generate_track(self):
        """
        Generate a closed loop track using noisy circle.
        """
        n_points = 200
        thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        radii = self.track_radius + 6.0 * np.sin(3 * thetas) + 3.0 * np.sin(7 * thetas)

        xs = radii * np.cos(thetas)
        ys = radii * np.sin(thetas)
        self.track = np.stack([xs, ys], axis=1).astype(np.float32)

        # cumulative length for progress measure
        diffs = np.diff(np.vstack([self.track, self.track[0]]), axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self.track_length = np.cumsum(seg_lengths)
        self.track_length[-1] = self.track_length[-2] + seg_lengths[-1]

    def world_to_image(self, p):
        """Map world coords to image coords."""
        scale = self.obs_size / (2.5 * self.track_radius)
        x = p[0] * scale + self.obs_size / 2
        y = p[1] * scale + self.obs_size / 2
        return np.array([x, y])

    def compute_nearest_track_point(self, pos):
        dists = np.linalg.norm(self.track - pos[None, :], axis=1)
        idx = int(np.argmin(dists))
        dist = float(dists[idx])
        return idx, dist

    # ------------- gym API -------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_internal()

        # Put car onto track start, heading along tangent
        start_idx = 0
        self.position = self.track[start_idx].copy()

        next_idx = (start_idx + 1) % len(self.track)
        direction = self.track[next_idx] - self.track[start_idx]
        self.heading = float(np.arctan2(direction[1], direction[0]))

        self.closest_idx = start_idx
        self.progress = 0.0

        obs = self.render()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        steer = float(np.clip(action[0], -1.0, 1.0))
        gas = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        # --- update heading ---
        self.heading += steer * 0.05  # steering sensitivity

        # --- update speed ---
        accel = gas * self.accel_power - brake * self.brake_power
        self.speed += accel
        self.speed *= self.friction
        self.speed = float(np.clip(self.speed, -self.max_speed, self.max_speed))

        # --- update position ---
        direction = np.array([np.cos(self.heading), np.sin(self.heading)], dtype=np.float32)
        self.position += direction * self.speed

        self.step_count += 1

        # --- compute distance & progress ---
        idx, dist_to_center = self.compute_nearest_track_point(self.position)

        # progress: if we move forward along track index, give reward
        forward_steps = (idx - self.closest_idx) % len(self.track)
        if forward_steps > len(self.track) / 2:
            forward_steps = 0  # avoid wrapping jump

        self.progress += forward_steps
        self.closest_idx = idx

        # --- reward ---
        reward = 0.05  # alive bonus

        reward += 0.1 * forward_steps
        if dist_to_center > self.track_width:
            reward -= 1.0  # off-track penalty

        # --- termination ---
        terminated = False
        truncated = False

        if dist_to_center > self.off_track_threshold:
            terminated = True
            reward -= 20.0
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.render()
        info = {
            "dist_to_center": dist_to_center,
            "progress": self.progress,
            "speed": self.speed,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        img = np.ones((self.obs_size, self.obs_size, 3), dtype=np.uint8) * 120

        # draw track (centerline & width)
        pts_world = self.track
        pts_img = np.array([self.world_to_image(p) for p in pts_world], dtype=np.int32)

        cv2.polylines(img, [pts_img], isClosed=True, color=(40, 40, 40), thickness=8)

        # draw inside lighter lane
        cv2.polylines(img, [pts_img], isClosed=True, color=(80, 80, 80), thickness=4)

        # draw car as triangle
        car_center = self.world_to_image(self.position)
        car_center = car_center.astype(np.int32)

        car_len = 6
        car_wid = 4
        yaw = self.heading

        front = car_center + np.array(
            [car_len * np.cos(yaw), car_len * np.sin(yaw)], dtype=np.float32
        )
        left = car_center + np.array(
            [car_wid * np.cos(yaw + 2.5), car_wid * np.sin(yaw + 2.5)], dtype=np.float32
        )
        right = car_center + np.array(
            [car_wid * np.cos(yaw - 2.5), car_wid * np.sin(yaw - 2.5)], dtype=np.float32
        )

        pts_car = np.stack([front, left, right], axis=0).astype(np.int32)
        cv2.fillConvexPoly(img, pts_car, (0, 0, 255))

        return img

    def close(self):
        # nothing special, but keep API compatible
        pass
