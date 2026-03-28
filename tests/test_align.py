"""
Basic smoke tests for body_align.

These tests verify module structure and option parsing without
requiring actual photos or ML model downloads.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_parse_bg_color():
    from body_align.align import _parse_bg_color
    bgr = _parse_bg_color("#1E1E1E")
    assert bgr == (30, 30, 30)  # BGR

    bgr = _parse_bg_color("#FF0000")
    assert bgr == (0, 0, 255)  # red in BGR

    bgr = _parse_bg_color("0000FF")
    assert bgr == (255, 0, 0)  # blue in BGR


def test_parse_size():
    from body_align.align import _parse_size
    w, h = _parse_size("800x1067")
    assert w == 800
    assert h == 1067

    w, h = _parse_size("1920x1080")
    assert w == 1920
    assert h == 1080


def test_module_exports():
    """Verify that the main public API is accessible."""
    from body_align import align_photos
    assert callable(align_photos)


def test_models_module():
    """Verify model helper functions exist."""
    from body_align.models import get_pose_model_path, get_segmentation_model_path
    assert callable(get_pose_model_path)
    assert callable(get_segmentation_model_path)


def test_cli_entry_point():
    """Verify CLI entry point is importable."""
    from body_align.cli import main
    assert callable(main)


def test_cli_no_args(tmp_path):
    """CLI should exit with error when no photos provided."""
    from click.testing import CliRunner
    from body_align.cli import main

    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0
    assert "no photos" in result.output.lower() or "error" in result.output.lower()


def test_cli_help():
    """CLI --help should succeed."""
    from click.testing import CliRunner
    from body_align.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "front" in result.output.lower()
    assert "back" in result.output.lower()


def test_align_photos_missing_files(tmp_path):
    """align_photos should handle missing input files gracefully."""
    from body_align.align import align_photos

    result = align_photos(
        front_path=str(tmp_path / "nonexistent_front.jpg"),
        back_path=str(tmp_path / "nonexistent_back.jpg"),
        profile_path=str(tmp_path / "nonexistent_profile.jpg"),
        output_dir=str(tmp_path / "output"),
        date="2026-01-01",
        options={"no_bg": True},
    )
    # Should return success=False since no files existed
    assert result["success"] is False
    assert result["date"] == "2026-01-01"


def test_cache_dir_location():
    """Verify model cache dir is in ~/.cache/body-align."""
    from body_align.models import CACHE_DIR
    from pathlib import Path
    assert CACHE_DIR == Path.home() / ".cache" / "body-align"
