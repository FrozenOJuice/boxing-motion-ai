"""Optional frame-local, validity-gated image-plane arm geometry; never 3D angles."""

from copy import deepcopy

import torch
from torch.utils.data import Dataset

LIMB_VERSION = "torso-arm-geometry-1.0"
GROUPS = ("shoulder_wrist", "upper_arm", "forearm", "elbow_cosine", "upper_torso", "forearm_torso")
MASK_COLUMNS = (2, 5, 8, 10, 13, 16)
REFLECT_COLUMNS = (0, 3, 6, 12, 15)
LIMB_POLICY = {
    "version": LIMB_VERSION,
    "layout": "65 unchanged XY/velocity channels + left/right 17-channel arm geometry",
    "side_layout": "wrist-shoulder XY,mask; elbow-shoulder XY,mask; wrist-elbow XY,mask; "
    "elbow cosine,mask; upper-arm torso dot/cross,mask; forearm torso dot/cross,mask",
    "mask_control": "16 existing validity bits + left/right six geometry-validity bits",
    "minimum_projected_segment_torso_units": 0.001,
    "invalid": "zero sentinel plus false feature mask; never imputation",
    "orientation": "unit hip-midpoint to shoulder-midpoint axis, no stance estimate",
    "angle": "cosine of projected shoulder-elbow-wrist angle; not a 3D anatomical angle",
    "temporal": "frame-local only; original backward velocity handling unchanged",
}


def limb_features(x):
    """Return [..., T, 34] geometry; never trust invalid source coordinates."""
    if x.ndim < 2 or x.shape[-1] != 65 or not torch.isfinite(x).all():
        raise ValueError("Expected finite temporal 65-channel XY+velocity inputs")
    joints = x[..., :40].reshape(*x.shape[:-1], 8, 5)
    valid = joints[..., 4].bool() & x[..., 40, None].bool()
    xy = torch.where(valid[..., None], joints[..., :2], 0)
    torso = (xy[..., 0, :] + xy[..., 1, :] - xy[..., 6, :] - xy[..., 7, :]) / 2
    eps = LIMB_POLICY["minimum_projected_segment_torso_units"]
    torso_norm = torch.linalg.vector_norm(torso, dim=-1)
    torso_ok = valid[..., [0, 1, 6, 7]].all(-1) & (torso_norm > eps)
    torso_unit = torso / torso_norm.clamp_min(eps)[..., None]
    sides = []
    for side in range(2):
        s, e, w = side, side + 2, side + 4
        upper, fore = xy[..., e, :] - xy[..., s, :], xy[..., w, :] - xy[..., e, :]
        wrist = xy[..., w, :] - xy[..., s, :]
        un, fn = upper.norm(dim=-1), fore.norm(dim=-1)
        up_ok = valid[..., s] & valid[..., e] & (un > eps)
        fore_ok = valid[..., e] & valid[..., w] & (fn > eps)
        wrist_ok = valid[..., s] & valid[..., w]
        u, f = upper / un.clamp_min(eps)[..., None], fore / fn.clamp_min(eps)[..., None]
        angle_ok = up_ok & fore_ok
        cosine = (-(u * f).sum(-1)).clamp(-1, 1)[..., None]

        def oriented(vector):
            dot = (torso_unit * vector).sum(-1)
            cross = torso_unit[..., 0] * vector[..., 1] - torso_unit[..., 1] * vector[..., 0]
            return torch.stack((dot, cross), -1)

        values = (wrist, upper, fore, cosine, oriented(u), oriented(f))
        masks = (wrist_ok, up_ok, fore_ok, angle_ok, up_ok & torso_ok, fore_ok & torso_ok)
        sides.append(
            torch.cat(
                [
                    part
                    for value, mask in zip(values, masks, strict=True)
                    for part in (
                        torch.where(mask[..., None], value, 0),
                        mask[..., None].to(x.dtype),
                    )
                ],
                -1,
            )
        )
    return torch.cat(sides, -1)


class LimbClips(Dataset):
    """Append geometry to immutable velocity clips; no selection or new source access."""

    def __init__(self, dataset):
        if dataset.provenance["recording_id"] not in ("V4", "V5", "V6", "V7", "V8", "V10"):
            raise ValueError("Geometry scope excludes reserved recordings")
        self.clips = []
        for c in dataset.clips:
            geometry = limb_features(c["x"])
            masks = geometry.reshape(len(geometry), 2, 17)[..., list(MASK_COLUMNS)].reshape(-1, 12)
            self.clips.append(
                {
                    **c,
                    "x": torch.cat((c["x"], geometry), -1),
                    "mask_x": torch.cat((c["mask_x"], masks), -1),
                }
            )
        self.provenance = deepcopy(dataset.provenance)
        self.provenance.update(limb_policy=LIMB_POLICY, feature_channels=99, mask_channels=28)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return self.clips[i]
