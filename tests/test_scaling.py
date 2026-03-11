"""Tests for tuna.scaling — scaling policy dataclasses and YAML loader."""

import tempfile

from tuna.scaling import SpotScaling, default_scaling_policy, load_scaling_policy


class TestSpotScalingDefaults:
    def test_downscale_delay_default_is_60(self):
        policy = SpotScaling()
        assert policy.downscale_delay == 60

    def test_upscale_delay_default_is_5(self):
        policy = SpotScaling()
        assert policy.upscale_delay == 5

    def test_default_scaling_policy_downscale_delay(self):
        policy = default_scaling_policy()
        assert policy.spot.downscale_delay == 60


class TestLoadScalingPolicyDownscaleDelay:
    def test_custom_downscale_delay_from_yaml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("spot:\n  downscale_delay: 30\n")
            f.flush()
            policy = load_scaling_policy(f.name)
        assert policy.spot.downscale_delay == 30

    def test_default_when_not_specified(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("spot:\n  target_qps: 20\n")
            f.flush()
            policy = load_scaling_policy(f.name)
        assert policy.spot.downscale_delay == 60
