"""
Kalman Filter for bounding box motion prediction.

State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
  cx, cy        — box center
  aspect_ratio  — width / height
  height        — box height
  vx, vy, va, vh — velocities of the above

Measurement vector: [cx, cy, aspect_ratio, height]

This is the standard SORT Kalman formulation (Bewley et al. 2016),
used unchanged in DeepSORT, StrongSORT, and BoostTrack++.
"""

import numpy as np


class KalmanFilter:
    """
    Constant-velocity Kalman filter for axis-aligned bounding boxes.

    The state space models each box as having constant velocity.
    Measurement noise and process noise are tuned to the SORT defaults
    which work well for pedestrian tracking at standard frame rates.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition matrix (constant velocity model)
        self._F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Measurement matrix (observe position only, not velocity)
        self._H = np.eye(ndim, 2 * ndim)

        # Motion/process uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(
        self, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from an initial measurement.

        Args:
            measurement: (cx, cy, aspect_ratio, height)

        Returns:
            Tuple of (mean, covariance) — initial state estimate.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict next state using the motion model.

        Args:
            mean:       Current state mean (8,).
            covariance: Current state covariance (8, 8).

        Returns:
            Predicted (mean, covariance).
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self._F @ mean
        covariance = self._F @ covariance @ self._F.T + Q
        return mean, covariance

    def project(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state distribution to measurement space.

        Args:
            mean:       State mean (8,).
            covariance: State covariance (8, 8).

        Returns:
            Projected (mean, covariance) in measurement space.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        R = np.diag(np.square(std))
        innovation_cov = self._H @ covariance @ self._H.T + R
        return self._H @ mean, innovation_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update state estimate with a new measurement.

        Args:
            mean:        Predicted state mean (8,).
            covariance:  Predicted state covariance (8, 8).
            measurement: Observed (cx, cy, aspect_ratio, height).

        Returns:
            Updated (mean, covariance).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Kalman gain
        K = covariance @ self._H.T @ np.linalg.inv(projected_cov)

        innovation = measurement - projected_mean
        new_mean = mean + K @ innovation
        new_cov = (np.eye(len(mean)) - K @ self._H) @ covariance
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Mahalanobis distance between state and measurements.

        Used to gate (reject) implausible associations before Hungarian matching.

        Args:
            mean:         State mean (8,).
            covariance:   State covariance (8, 8).
            measurements: Array of (N, 4) measurements.

        Returns:
            Array of (N,) Mahalanobis distances.
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        d = measurements - projected_mean
        # Squared Mahalanobis distance
        inv_cov = np.linalg.inv(projected_cov)
        distances = np.sum(d @ inv_cov * d, axis=1)
        return distances
