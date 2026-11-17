# VisualQC — Workshop worksheet

The [README](README.md) explains setup and the rules. This sheet is the plan for the whole day: what to do, in what order, and what evidence to collect. Times are suggestions; your instructor may move them.

**Daily theme:** AI for Software Engineering and Data — Coding, Analysis and Intelligent Workflows.

## How to work today

- Work in your team on a **branch** of your fork. Commit often.
- Keep a lab notebook in `NOTES.md` at the repo root and commit it. It is part of your submission:

  ```markdown
  # Lab notebook — team <name>
  - Tool / model(s):
  - Prompt log (one line per important prompt):
  - AI suggestions: accepted / modified / rejected, and why
  - Evidence: commands you ran + the key numbers
  - AI mistakes or unsupported claims (at least 1)
  - Limitations of our solution (at least 1)
  ```
- **"The AI said so" is not evidence.** Evidence is a command and its output, a line of code, a plot, or a documentation link.
- Your coding agent must follow the README rules. Optional but recommended: put the rules in `AGENTS.md` (OpenCode's `/init` creates one) and commit it. `tests/test_rules.py` checks the most important rules automatically. It is a guard, not a sandbox, so a determined agent can still get around it. That is why you review its diffs.

---

## Block 1 · 09:15–11:30

### Setup (09:15–09:30)

Follow the README quick start. Expected result of `uv run pytest -q`: **exactly one failing test** (`test_detector_finds_at_least_something_on_known_defect`). Everything else passes, including `tests/test_rules.py`.

### Part 1 · AI-assisted software development (09:30–10:20)

1. **Plan before code.** Ask your agent to read `TASK.md` and `visual_qc/detector.py` and propose a pipeline *before* writing code. Copy the plan into `NOTES.md`. Which assumptions did it make about the images?
2. **Generate.** Let it implement `detect_defects()`. Run `uv run pytest -q` and `uv run python -m visual_qc.evaluate --split train --out results/train`.
3. **Debug.** Find the worst train image in `results/train/metrics.csv`. Ask the agent *why* it fails. Then check the explanation yourself: compare `results/train/predictions/<id>.png` with the image and its mask. Was the explanation right?
4. **Test.** Ask the agent to add one new test in `tests/test_detector_extra.py` for a case it thinks is hard. Example: a synthetic image with a strong lighting gradient and no defect should give an empty mask. Then break the detector on purpose (e.g. return all 255). Does the new test catch it? A test that never fails is not a test.
5. **Refactor and document.** Ask the agent to split `detect_defects()` into small named steps with docstrings, *without changing behaviour*. Prove it: `metrics.csv` before and after must be identical per image.

### Part 2 · Data + AI: from metrics to features (10:20–11:20)

1. **Run** the evaluator on `train` and on `test`, and make the summary plot (README step 5).
2. **Feature thinking.** Write `analysis/features.py`, **outside** `visual_qc/`. Reading masks is allowed here, because this is analysis, not the detector. For every image, compute features of its reference defects:
   - defect area (pixels) and number of connected components
   - elongation (scratch vs. spot), e.g. from `cv2.minAreaRect`
   - contrast: mean intensity inside the defect minus mean intensity in a thin ring around it (negative = dark, positive = bright)
   - background gradient: intensity range of a heavily blurred image

   Join the features with `metrics.csv` on `id`. Save the result to `results/features_metrics.csv`.
3. **Which feature best explains low IoU?** Make one plot that shows it. Ask the AI to interpret your plot. Then mark every sentence of its interpretation as *supported*, *not supported* or *not checkable* with your data.
4. **Rules of evidence.**
   - Tune thresholds on `train` only, and report `test` once at the end.
   - `test` has only 18 images (15 defective, 3 intact). How confident can a conclusion from 18 images be?
   - Read `visual_qc/evaluation.py`: what IoU does an intact image get with an empty prediction? With one stray pixel? Is *mean IoU* a fair single number for this dataset?

### Wrap-up (11:20–11:30)

Commit, push your branch, and update `NOTES.md`.

**Fast team? Stretch goals:**
- handle dark and bright defects separately
- measure the accuracy–latency trade-off for 3 settings of your pipeline
- the optional PyTorch extension (README)

---

## Block 2 · 12:00–13:40

### Part 3 · Case-study lab: compare AI outputs (12:00–12:25)

1. **Results card.** Fill in your team's row and share it where your instructor shows you:

   | Team | Tool | Model | # prompts | Train IoU | Test IoU | Median latency (ms) | `test_rules.py` green first time? | Suggestions accepted / modified / rejected |
   |---|---|---|---|---|---|---|---|---|

2. **Compare across teams.** Everyone had the same task and the same evaluator, but different models. What differs: the approach, the quality, the number of iterations? How often did an agent claim an improvement it had not measured?
3. **Usefulness.** Pick 3 AI suggestions. For each, estimate time saved vs. time spent verifying or fixing it. Was the AI net-positive?

### Part 4 · Detecting hallucinations + human-in-the-loop verification (12:25–12:50)

1. **Claim check.** For each claim, decide **True**, **False** or **Unsupported** (not decidable from the available evidence), and give the evidence. Run code, read the source, read the docs.

   | # | Claim |
   |---|---|
   | 1 | "Otsu thresholding on the grayscale image handles the uneven lighting in these images." |
   | 2 | "Adding CLAHE before thresholding will improve IoU on this dataset." |
   | 3 | "`cv2.imread` returns images in RGB channel order." |
   | 4 | "`cv2.medianBlur` accepts `float32` images with `ksize=31`." |
   | 5 | "An empty prediction on an intact image gets IoU = 1.0 in this evaluation." |
   | 6 | "Our median latency of X ms means the detector can process 1000/X images per second in production." |
   | 7 | "Mean IoU on the test split is an unbiased estimate of performance on a real production line." |
   | 8 | "A morphological opening with a 3×3 kernel removes noise but keeps the scratches in this dataset." |

2. **Your own agent.** Take 2 claims your agent made today. Treat them the same way and record them in `NOTES.md`.
3. **Agent audit.** Review what your agent changed: `git diff main --stat`, then the full diff.
   - Did it touch `tests/`, `visual_qc/evaluation.py`, `visual_qc/evaluate.py` or `data/`?
   - Did it read `data/masks` while developing? Check its session log.
   - Is `tests/test_rules.py` green?
   - Bonus: find one way around `test_rules.py`, explain how a human reviewer would catch it, and **do not commit it**.

### Part 5 · Team project: architecture and workflow design (12:50–13:25)

Pick **one** AI Engineering Challenge. Every challenge delivers the same five things (in `NOTES.md` or `docs/`):

1. an **architecture diagram** (Mermaid in Markdown, or a photo of a paper sketch)
2. the **workflow**: where AI generates, where a human verifies, and which automated check gates what
3. a **trade-offs** table comparing at least 2 options
4. **one measured result**
5. **one documented limitation**

**Challenge A — Better detector.** Use what Part 2 told you to improve the classical pipeline. Show results tuned on train and reported on test, plus the accuracy–latency trade-off.

**Challenge B — Human-in-the-loop QC service.** Design the detector as a service:
- **API contract.** Request: image. Response: mask, defect area, a `needs_review` flag and the reason. Decide between batch and single-image calls, and set a latency budget.
- **Review rule.** Define a rule that sends an image to a human, e.g. defect area close to your decision threshold, or many tiny components. Measure: if a human reviews the top-k flagged test images, how many of the worst-IoU images are caught?
- **Automation.** Add a CI quality gate, e.g. fail the build if mean train IoU drops below your baseline. Sketch it or implement it.

**Challenge C — Retrieval-based approaches.** Choose one:
- **C1 · Retrieval for the data.** For a new image, find the k most similar *train* images using simple features (downsampled grayscale, brightness histogram, gradient strength), and reuse the thresholds tuned for them. Compare with one global threshold. The detector may load a small precomputed index file stored in `visual_qc/`, but it may still not read `data/masks` or `dataset.csv`.
- **C2 · Retrieval for the assistant.** Give your agent the relevant OpenCV documentation pages, which you retrieve yourself, as context. Re-ask the Part 4 claims, and compare the answers with and without the docs.

### Presentations (13:25–13:40)

3 minutes per team: one diagram, one number, one limitation, and one AI mistake you caught.

## Submission

Open a pull request **to your own fork** (README, "GitHub submission"). CI runs the same tests, including `tests/test_rules.py`. Include `NOTES.md`, your analysis script and your diagram.
