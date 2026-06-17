"""Unit tests for app utility functions."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import feature_label, safe_text


class TestFeatureLabel:
    """Test audio feature z-score to label conversion."""
    
    def test_high_feature_value(self) -> None:
        """Test high z-score returns 'high' label."""
        result = feature_label(1.5, "high energy", "low energy", "balanced")
        assert result == "high energy"
    
    def test_low_feature_value(self) -> None:
        """Test low z-score returns 'low' label."""
        result = feature_label(-1.5, "high energy", "low energy", "balanced")
        assert result == "low energy"
    
    def test_neutral_feature_value(self) -> None:
        """Test z-score near 0 returns neutral label."""
        result = feature_label(0.3, "high energy", "low energy", "balanced")
        assert result == "balanced"
    
    def test_boundary_high(self) -> None:
        """Test boundary at 0.65 switches to high."""
        result_below = feature_label(0.64, "high", "low", "neutral")
        result_at = feature_label(0.65, "high", "low", "neutral")
        
        assert result_below == "neutral"
        assert result_at == "high"
    
    def test_boundary_low(self) -> None:
        """Test boundary at -0.65 switches to low."""
        result_above = feature_label(-0.64, "high", "low", "neutral")
        result_at = feature_label(-0.65, "high", "low", "neutral")
        
        assert result_above == "neutral"
        assert result_at == "low"


class TestSafeText:
    """Test HTML escaping for safe output."""
    
    def test_escape_html_chars(self) -> None:
        """Test HTML special characters are escaped."""
        assert safe_text("hello & world") == "hello &amp; world"
        assert safe_text("<script>") == "&lt;script&gt;"
        assert safe_text('say "hi"') == "say &quot;hi&quot;"
    
    def test_safe_text_none(self) -> None:
        """Test None returns empty string."""
        assert safe_text(None) == ""
    
    def test_safe_text_normal_string(self) -> None:
        """Test normal strings pass through unchanged."""
        assert safe_text("normal text") == "normal text"
        assert safe_text("Track Name") == "Track Name"
    
    def test_safe_text_numbers(self) -> None:
        """Test numeric values are converted safely."""
        assert safe_text(42) == "42"
        assert safe_text(3.14) == "3.14"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
