"""Replaceable synchronous MediaPipe backend; no future frames or gap interpolation."""


class MediaPipePose:
    def __init__(self, model, confidence=0.5, mode="image"):
        import mediapipe as mp

        if mode not in {"image", "video"}:
            raise ValueError("Pose mode must be image or video")
        self.mode = mode
        self.mp = mp
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=getattr(mp.tasks.vision.RunningMode, mode.upper()),
            num_poses=1,
            min_pose_detection_confidence=confidence,
            min_pose_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            output_segmentation_masks=False,
        )
        self.detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def detect(self, rgb, timestamp_ms):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = (
            self.detector.detect(image)
            if self.mode == "image"
            else self.detector.detect_for_video(image, timestamp_ms)
        )
        if not result.pose_landmarks:
            return None
        return [(p.x, p.y, p.z, p.visibility, p.presence) for p in result.pose_landmarks[0]]

    def close(self):
        self.detector.close()
