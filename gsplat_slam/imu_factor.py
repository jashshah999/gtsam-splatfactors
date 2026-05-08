"""IMU preintegration factor for multi-sensor fusion.

This is the key differentiator over SplaTAM/MonoGS — by having poses in
a GTSAM factor graph, we can trivially add IMU preintegration factors
alongside photometric factors. IMU provides:
- Metric scale (resolves monocular scale ambiguity)
- Drift reduction between keyframes
- Robustness during fast motion / blur (where photometric fails)
"""

import numpy as np

try:
    import gtsam
    HAS_GTSAM = True
except ImportError:
    HAS_GTSAM = False


class IMUPreintegrator:
    """Preintegrates IMU measurements between keyframes for GTSAM factors."""

    def __init__(
        self,
        accel_noise_sigma: float = 0.1,
        gyro_noise_sigma: float = 0.01,
        accel_bias_sigma: float = 0.001,
        gyro_bias_sigma: float = 0.0001,
        gravity: np.ndarray = None,
    ):
        assert HAS_GTSAM, "GTSAM required for IMU factors"

        if gravity is None:
            gravity = np.array([0, 0, -9.81])

        # IMU noise parameters
        measured_acc_cov = np.eye(3) * accel_noise_sigma ** 2
        measured_omega_cov = np.eye(3) * gyro_noise_sigma ** 2
        integration_error_cov = np.eye(3) * 1e-8

        self.params = gtsam.PreintegrationParams.MakeSharedU(gravity)
        self.params.setAccelerometerCovariance(measured_acc_cov)
        self.params.setGyroscopeCovariance(measured_omega_cov)
        self.params.setIntegrationCovariance(integration_error_cov)

        # Bias noise model
        self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(
            6, max(accel_bias_sigma, gyro_bias_sigma)
        )

        self.current_bias = gtsam.imuBias.ConstantBias()
        self.preintegrated = gtsam.PreintegratedImuMeasurements(
            self.params, self.current_bias
        )
        self.measurement_count = 0

    def add_measurement(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Add a single IMU measurement.

        Args:
            accel: (3,) accelerometer reading [m/s^2]
            gyro: (3,) gyroscope reading [rad/s]
            dt: Time delta since last measurement [s]
        """
        self.preintegrated.integrateMeasurement(accel, gyro, dt)
        self.measurement_count += 1

    def add_measurements_batch(self, accels: np.ndarray, gyros: np.ndarray, timestamps: np.ndarray):
        """Add a batch of IMU measurements.

        Args:
            accels: (N, 3) accelerometer readings
            gyros: (N, 3) gyroscope readings
            timestamps: (N,) timestamps in seconds
        """
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > 0:
                self.add_measurement(accels[i], gyros[i], dt)

    def create_factor(self, key_i: int, key_j: int, vel_key_i: int, vel_key_j: int, bias_key: int):
        """Create a GTSAM ImuFactor from preintegrated measurements.

        Args:
            key_i: GTSAM key for pose at time i
            key_j: GTSAM key for pose at time j
            vel_key_i: GTSAM key for velocity at time i
            vel_key_j: GTSAM key for velocity at time j
            bias_key: GTSAM key for IMU bias

        Returns:
            gtsam.ImuFactor
        """
        return gtsam.ImuFactor(
            key_i, vel_key_i,
            key_j, vel_key_j,
            bias_key,
            self.preintegrated,
        )

    def reset(self, new_bias: "gtsam.imuBias.ConstantBias | None" = None):
        """Reset preintegration for next keyframe interval."""
        if new_bias is not None:
            self.current_bias = new_bias
        self.preintegrated = gtsam.PreintegratedImuMeasurements(
            self.params, self.current_bias
        )
        self.measurement_count = 0

    @property
    def predicted_nav_state(self):
        """Get predicted navigation state from preintegration."""
        return self.preintegrated.predict(
            gtsam.NavState(gtsam.Pose3(), np.zeros(3)),
            self.current_bias,
        )


def add_imu_to_slam(
    slam,
    imu_data: dict,
    keyframe_timestamps: list[float],
):
    """Add IMU factors to an existing SplatSLAM instance.

    Args:
        slam: SplatSLAM instance with poses already estimated
        imu_data: dict with 'accels' (N,3), 'gyros' (N,3), 'timestamps' (N,)
        keyframe_timestamps: timestamps for each keyframe in slam
    """
    accels = imu_data["accels"]
    gyros = imu_data["gyros"]
    imu_timestamps = imu_data["timestamps"]

    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    preintegrator = IMUPreintegrator()
    n_kf = len(keyframe_timestamps)

    for i in range(n_kf - 1):
        t_start = keyframe_timestamps[i]
        t_end = keyframe_timestamps[i + 1]

        # Get IMU measurements between these keyframes
        mask = (imu_timestamps >= t_start) & (imu_timestamps < t_end)
        if mask.sum() < 2:
            continue

        preintegrator.reset()
        preintegrator.add_measurements_batch(
            accels[mask], gyros[mask], imu_timestamps[mask]
        )

        # Create IMU factor
        pose_key_i = gtsam.symbol("x", i)
        pose_key_j = gtsam.symbol("x", i + 1)
        vel_key_i = gtsam.symbol("v", i)
        vel_key_j = gtsam.symbol("v", i + 1)
        bias_key = gtsam.symbol("b", 0)

        factor = preintegrator.create_factor(
            pose_key_i, pose_key_j, vel_key_i, vel_key_j, bias_key
        )
        graph.add(factor)

        # Add velocity priors if not already in graph
        if not slam.isam2.valueExists(vel_key_i):
            values.insert(vel_key_i, np.zeros(3))
        if not slam.isam2.valueExists(vel_key_j):
            values.insert(vel_key_j, np.zeros(3))

    # Add bias prior
    bias_key = gtsam.symbol("b", 0)
    if not slam.isam2.valueExists(bias_key):
        values.insert(bias_key, gtsam.imuBias.ConstantBias())
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.001)
        graph.addPriorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise)

    slam.isam2.update(graph, values)
    print(f"  Added {n_kf - 1} IMU preintegration factors")
