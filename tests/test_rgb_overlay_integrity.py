"""Pixel-integrity checks: mask only observed pixels; never fabricate obscured motion."""

import numpy as np
import pytest
from PIL import Image
from test_video_foundation import raw

from boxing_motion_ai.vision.rgb_overlay_integrity import discs, masks, process


def test_targeted_footer_preserves_blue_pixels_but_cannot_distinguish_white_foot_from_text():
    rgb = np.full((100, 200, 3), [50, 80, 160], np.uint8)
    rgb[90:95, 40:50] = 255  # Could be text or a white shoe detail: same observed pixels.
    rgb[30:35, 40:50] = 255  # Glove outside the fixed footer.
    image = Image.fromarray(rgb)
    original = rgb.copy()
    keep = masks(image, np.ones((100, 200)), raw(1), 0, "V6")
    target = keep["targeted_bright_ink"]
    full = keep["whole_footer_removed"]
    assert target[85, 45] and not target[92, 45] and target[32, 45]
    assert not full[85, 45] and full[32, 45]
    output = np.asarray(process(image, target))
    assert np.array_equal(output[target], rgb[target])
    assert (output[~target] == 128).all()
    assert np.array_equal(np.asarray(image), original)


def test_person_mask_missing_does_not_fall_back_to_leaked_source():
    image = Image.fromarray(np.full((100, 200, 3), 60, np.uint8))
    views = masks(image, np.zeros((100, 200)), raw(1), 0, "V6")
    assert all(not m.any() for m in views.values())
    for source in ("V5", "V7", "V9"):
        with pytest.raises(Exception, match="rgb_overlay_forbidden_source"):
            masks(image, np.ones((100, 200)), raw(1), 0, source)
    with pytest.raises(Exception, match="rgb_overlay_mask_shape"):
        masks(image, np.ones((5, 5)), raw(1), 0, "V6")


def test_disc_metric_excludes_missing_and_offscreen_observations():
    r = raw(1)
    r["valid_mask"][0, 15] = False
    r["coordinates"][0, 16, 0] = 1.2
    assert not discs(r, 0, (200, 100), (15, 16)).any()
    assert discs(r, 0, (200, 100), (31, 32)).any()
