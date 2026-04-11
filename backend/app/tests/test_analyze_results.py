from app.services.interpretation.analyze_results import (
    extract_feature_importance,
    paired_significance_test,
)


def test_paired_significance_test_identical_values():
    """Identical fold metrics should produce p=1.0 (no difference)."""
    result = paired_significance_test([0.7, 0.8, 0.75, 0.72, 0.78], [0.7, 0.8, 0.75, 0.72, 0.78])
    assert result["p_value"] == 1.0
    assert result["mean_difference"] == 0.0
    assert result["significant_at_005"] is False


def test_paired_significance_test_clear_difference():
    """Large consistent difference with some variance should be significant."""
    result = paired_significance_test([0.90, 0.88, 0.92, 0.87, 0.91], [0.60, 0.55, 0.62, 0.58, 0.61])
    assert result["p_value"] < 0.01
    assert result["significant_at_005"] is True
    assert result["mean_difference"] > 0.25


def test_paired_significance_test_returns_all_fields():
    result = paired_significance_test([0.8, 0.7], [0.6, 0.5])
    assert "t_statistic" in result
    assert "p_value" in result
    assert "mean_difference" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert "significant_at_005" in result
    assert "n_folds" in result
    assert result["n_folds"] == 2
