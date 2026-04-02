"""
Basic tests for pure functions in feedback_analyzer.py.
Run with: python3 -m pytest tests/test_feedback_analyzer.py -v
"""
import sys
import os
import math

# Allow importing from submission/src without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'submission', 'src'))

from feedback_analyzer import calculate_angle, calculate_symmetry_score, calculate_consistency_score


def _make_point(x, y, z=0.0):
    """Helper: create a landmark-style dict."""
    return {'x': x, 'y': y, 'z': z}


class TestCalculateAngle:
    """Tests for calculate_angle(p1, p2, p3)."""

    def test_right_angle(self):
        """Three points forming a 90-degree angle at p2."""
        p1 = _make_point(0.0, 1.0)
        p2 = _make_point(0.0, 0.0)
        p3 = _make_point(1.0, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 90.0) < 1e-4, f"Expected 90 degrees, got {angle}"

    def test_straight_line_180_degrees(self):
        """Three collinear points should yield 180 degrees."""
        p1 = _make_point(-1.0, 0.0)
        p2 = _make_point(0.0, 0.0)
        p3 = _make_point(1.0, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 180.0) < 1e-4, f"Expected 180 degrees, got {angle}"

    def test_zero_magnitude_returns_zero(self):
        """When p1 == p2 the magnitude is zero; function must return 0 and not raise."""
        p1 = _make_point(0.0, 0.0)  # Same as p2
        p2 = _make_point(0.0, 0.0)
        p3 = _make_point(1.0, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert angle == 0.0, f"Expected 0.0 for zero-magnitude vector, got {angle}"

    def test_known_45_degree_angle(self):
        """Vectors (1,0,0) and (1,1,0) from origin should give 45 degrees."""
        p1 = _make_point(1.0, 0.0)
        p2 = _make_point(0.0, 0.0)
        p3 = _make_point(1.0, 1.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 45.0) < 1e-4, f"Expected 45 degrees, got {angle}"

    def test_result_in_valid_range(self):
        """Angle must always be between 0 and 180 inclusive."""
        p1 = _make_point(0.3, 0.7)
        p2 = _make_point(0.5, 0.5)
        p3 = _make_point(0.8, 0.2)
        angle = calculate_angle(p1, p2, p3)
        assert 0.0 <= angle <= 180.0, f"Angle {angle} is out of the expected [0, 180] range"


class TestCalculateSymmetryScore:
    """Tests for calculate_symmetry_score(left_angle, right_angle)."""

    def test_identical_angles_score_ten(self):
        score = calculate_symmetry_score(90.0, 90.0)
        assert score == 10.0

    def test_large_difference_clamps_to_zero(self):
        score = calculate_symmetry_score(0.0, 180.0)
        assert score == 0.0

    def test_score_within_threshold_is_ten(self):
        score = calculate_symmetry_score(90.0, 93.0)  # diff=3, threshold=5
        assert score == 10.0


class TestCalculateConsistencyScore:
    """Tests for calculate_consistency_score(angles)."""

    def test_empty_list_returns_zero(self):
        score = calculate_consistency_score([])
        assert score == 0.0

    def test_single_element_returns_zero(self):
        score = calculate_consistency_score([90.0])
        assert score == 0.0

    def test_constant_angles_score_ten(self):
        score = calculate_consistency_score([90.0, 90.0, 90.0])
        assert score == 10.0

    def test_high_variance_lowers_score(self):
        score = calculate_consistency_score([0.0, 90.0, 180.0, 45.0, 135.0])
        assert score < 10.0
