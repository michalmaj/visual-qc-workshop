# VisualQC Workshop — AI-assisted Computer Vision & Data

**BIP workshop · 17 November 2026 · Python 3.11+ · OpenCV · uv**

Build a small computer-vision quality-control system with an AI coding assistant. Then **verify** whether it works and **measure** its limitations. The goal is not to write the most code; it is to make evidence-based engineering decisions.

## Quick start (Linux/macOS)

1. Download or fork this repository.
2. If `uv` is missing, install from the official source (requires HTTPS access):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Open a new terminal (or reload your shell) and verify `uv --version`.
3. From the repo root:
   ```bash
   uv sync
   uv run python -c "import cv2, numpy, pandas; print('Environment OK', cv2.__version__)"
   uv run pytest -q
   ```
   **Expected at the start:** one test deliberately fails because `detect_defects()` is a placeholder.
4. Open `visual_qc/detector.py` and complete the task with your AI coding assistant.
5. Run the independent evaluator and produce results:
   ```bash
   uv run pytest -q
   uv run python -m visual_qc.evaluate --split test --out results/test
   uv run python -m visual_qc.plot --csv results/test/metrics.csv --out results/test/summary.png
   ```

All source images and reference masks are **already in the repo**. No external dataset, downloaded ML model, GPU, or PyTorch installation is needed.

## Today's challenge

**The full day plan, with timings, extra tasks and the team challenges, is in [WORKSHEET.md](WORKSHEET.md).**

Images depict simple synthetic inspection scenes with scratches, spots, faint marks, bright marks, and variable lighting. Your function returns a binary segmentation mask (255 = defect). Evaluation computes per-image pixel IoU, F1, precision, recall, and per-image processing time.

| Stage | Your task | Evidence to produce |
| --- | --- | --- |
| 1. AI-assisted coding | Implement the segmentation function; ask AI to explain the approach | Git diff, working tests |
| 2. Data + AI | Run evaluator and plots; inspect poor-performing images | CSV and PNG plots |
| 3. Reliable workflow | Ask AI for a code review; verify claims against tests and docs | One documented AI mistake or unsupported claim |
| 4. Architecture | Propose a better implementation or learning-based extension | Small diagram + trade-offs |

## Prompt ideas

- “Inspect the contract in `visual_qc/detector.py` and propose a simple OpenCV pipeline. Explain your assumptions before writing code.”
- “Implement a deterministic segmentation algorithm using only the image argument, without reading any masks or metadata.”
- “Our IoU is poor on some samples. Suggest experiments, but don't claim improvements until we measure them.”
- “Audit my implementation for input mutation, incorrect colour-space assumptions, thresholds, and failure cases.”

## Important engineering rules

- **Do not read `data/masks` or `dataset.csv` from the detector.** Reference masks are for evaluation only.
- Do not make API requests or download weights from your detector.
- Do not modify the evaluation code to inflate scores.
- Report failures, not just successful examples.
- Timing on a shared workstation is **exploratory** and depends on CPU load, scheduling, and data size.
- Test on samples not used to set your thresholds. Public test masks are provided for the exercise, so this is a workshop evaluation, not a blind benchmark.
- `opencv-python-headless` is intentional: visualize with saved files, not `cv2.imshow()`.
- The first two rules, plus deterministic output, are checked by `tests/test_rules.py` (locally and in CI). It is a guard, not a sandbox. Review your agent's diffs, including any change to the evaluation code.

## Optional PyTorch extension (ONLY if already installed and working)

Try a small neural-network approach and compare quality and runtime on the same test images. CPU-only is enough for these tiny images, but PyTorch is not needed for the core workshop. Do not spend workshop time debugging CUDA or downloading pretrained weights.

## GitHub submission

Teams can fork the repository, work on a branch, and open a pull request **to their own fork**. GitHub Actions will run the same `pytest` checks. If the fork's **Actions** tab says workflows are disabled, enable them first. Keep a note of what the AI suggested, which checks you ran, and one meaningful limitation you found.
