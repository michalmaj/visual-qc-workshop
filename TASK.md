# Assignment: Detect visual defects with AI assistance

Implement `visual_qc.detector.detect_defects(image_bgr: np.ndarray) -> np.ndarray`.

- Input: uint8 BGR image, shape (H, W, 3).
- Output: uint8 binary mask, shape (H, W), with pixel values {0, 255} only.
- Handle intact images as well as images containing dark/bright spots and scratches.
- The solution must use only the input image; do not consult ground-truth masks or data labels inside the detector.
- Discuss what happens when illumination varies, defects are thin, or image noise is mistaken for a defect.
- Review accuracy AND time, and document at least one incorrect or unsupported claim made by the AI assistant.

**Suggested steps:** inspect examples -> ask AI for a plan -> implement -> run contract tests -> evaluate -> inspect failures -> refine -> rerun -> explain the evidence.

The initial failing test is intentional: it is the coding challenge, not a broken starter.
