from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration parameters for Kneesens accelerometer simulation."""

    sampling_rate_hz: int = 100
    walking_duration_s: float = 12.0
    fall_duration_s: float = 12.0
    gait_frequency_hz: float = 1.8
    noise_std_g: float = 0.03
    fall_time_s: float = 6.0
    impact_duration_s: float = 0.12
    random_seed: int = 42
    output_csv: str = "kneesens_fallback_data.csv"


class KneesensSimulator:
    """
    Simulate raw 3-axis accelerometer data for a knee-mounted fall-detection wearable.

    The simulator produces two labeled scenarios:
    1. Normal Walking
    2. Sudden Fall

    Modeling assumptions:
    - Walking is represented by low-amplitude periodic motion with a dominant gait frequency.
    - The vertical axis (Z) is centered near 1 g because gravity is always measured by
      the accelerometer in static or quasi-static conditions.
    - A fall is modeled as:
        gait -> short high-amplitude impact spike -> prolonged inactivity
    - White Gaussian noise is added to reflect low-cost sensor imperfections.
    """

    GRAVITY_MPS2: float = 9.80665

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self._validate_config()
        self.rng = np.random.default_rng(self.config.random_seed)

    def _validate_config(self) -> None:
        """Validate simulation parameters before generation."""
        cfg = self.config

        if cfg.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be > 0.")
        if cfg.walking_duration_s <= 0 or cfg.fall_duration_s <= 0:
            raise ValueError("Scenario durations must be > 0 seconds.")
        if cfg.gait_frequency_hz <= 0:
            raise ValueError("gait_frequency_hz must be > 0.")
        if cfg.noise_std_g < 0:
            raise ValueError("noise_std_g must be >= 0.")
        if cfg.impact_duration_s <= 0:
            raise ValueError("impact_duration_s must be > 0.")
        if not (0.0 < cfg.fall_time_s < cfg.fall_duration_s):
            raise ValueError("fall_time_s must lie within the fall scenario duration.")

    def _create_time_vector(self, duration_s: float) -> np.ndarray:
        """Create an evenly sampled time base."""
        sample_count = int(duration_s * self.config.sampling_rate_hz)
        return np.arange(sample_count, dtype=np.float64) / self.config.sampling_rate_hz

    def _simulate_walking_waveform(self, time_s: np.ndarray) -> np.ndarray:
        """
        Simulate normal walking in gravitational units.

        Mathematical idea:
        - Human gait is quasi-periodic, not perfectly sinusoidal.
        - We use a fundamental gait component plus a weaker harmonic.
        - The knee experiences oscillatory motion in all three axes.
        - Z is biased around 1 g to represent gravity.

        Returns:
            np.ndarray of shape (N, 3) with columns [X, Y, Z] in g.
        """
        f = self.config.gait_frequency_hz
        omega = 2.0 * np.pi * f

        accel_x = (
            0.12 * np.sin(omega * time_s)
            + 0.03 * np.sin(2.0 * omega * time_s + 0.40)
        )
        accel_y = (
            0.08 * np.sin(omega * time_s + np.pi / 4.0)
            + 0.02 * np.sin(2.0 * omega * time_s + 1.10)
        )
        accel_z = (
            1.00
            + 0.20 * np.sin(omega * time_s + np.pi / 2.0)
            + 0.04 * np.sin(2.0 * omega * time_s + 0.15)
        )

        return np.column_stack((accel_x, accel_y, accel_z))

    def _simulate_sudden_fall(self, time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate a sudden fall scenario in gravitational units.

        Fall model:
        1. Pre-fall: normal gait
        2. Impact: narrow Gaussian pulse to represent abrupt collision/deceleration
        3. Post-fall: minimal dynamic movement with the body lying still

        The Gaussian pulse is used because it gives a compact, realistic impact profile
        with a sharp peak and smooth rise/fall, avoiding a numerically harsh step impulse.

        Returns:
            Tuple of:
            - accelerometer array with shape (N, 3) in g
            - state labels with shape (N,)
        """
        base_signal = self._simulate_walking_waveform(time_s)
        signal = base_signal.copy()

        fall_time = self.config.fall_time_s
        impact_sigma = self.config.impact_duration_s / 6.0

        impact_pulse = np.exp(-0.5 * ((time_s - fall_time) / impact_sigma) ** 2)

        # Axis-specific pulse amplitudes create a visibly strong multi-axis impact.
        signal[:, 0] += 2.8 * impact_pulse
        signal[:, 1] += -2.1 * impact_pulse
        signal[:, 2] += 3.6 * impact_pulse

        states = np.full(time_s.shape, "pre_fall_gait", dtype=object)

        impact_window = np.abs(time_s - fall_time) <= (self.config.impact_duration_s / 2.0)
        states[impact_window] = "impact"

        inactivity_start = fall_time + self.config.impact_duration_s
        inactivity_mask = time_s >= inactivity_start

        # After impact, dynamic movement is close to zero while gravity remains present.
        settle_time = time_s[inactivity_mask] - inactivity_start
        low_drift = 0.004 * np.sin(2.0 * np.pi * 0.20 * settle_time)

        signal[inactivity_mask, 0] = 0.00 + low_drift
        signal[inactivity_mask, 1] = 0.00 + 0.50 * low_drift
        signal[inactivity_mask, 2] = 1.00 + 0.25 * low_drift
        states[inactivity_mask] = "inactive_lying"

        return signal, states

    def _add_sensor_noise(self, accel_g: np.ndarray) -> np.ndarray:
        """
        Add white Gaussian noise to simulate low-cost MEMS sensor behavior.

        Noise sources represented here include quantization-like jitter, electronic noise,
        and mechanical imperfections commonly seen in inexpensive wearable IMUs.
        """
        noise = self.rng.normal(
            loc=0.0,
            scale=self.config.noise_std_g,
            size=accel_g.shape,
        )
        return accel_g + noise

    def _build_dataframe(
        self,
        scenario: str,
        state: np.ndarray,
        time_s: np.ndarray,
        accel_g: np.ndarray,
    ) -> pd.DataFrame:
        """Convert simulated data into a structured pandas DataFrame."""
        df = pd.DataFrame(
            {
                "Scenario": scenario,
                "State": state,
                "Sample_Index": np.arange(len(time_s), dtype=np.int64),
                "Time_s": time_s,
                "Accel_X_g": accel_g[:, 0],
                "Accel_Y_g": accel_g[:, 1],
                "Accel_Z_g": accel_g[:, 2],
            }
        )

        df["Accel_X_mps2"] = df["Accel_X_g"] * self.GRAVITY_MPS2
        df["Accel_Y_mps2"] = df["Accel_Y_g"] * self.GRAVITY_MPS2
        df["Accel_Z_mps2"] = df["Accel_Z_g"] * self.GRAVITY_MPS2

        df["Accel_Magnitude_g"] = np.sqrt(
            df["Accel_X_g"] ** 2 + df["Accel_Y_g"] ** 2 + df["Accel_Z_g"] ** 2
        )

        return df
