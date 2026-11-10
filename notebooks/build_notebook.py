"""Generate notebooks/colab_train.ipynb. Edit this file, not the .ipynb, then re-run it:

    .venv/bin/python -m notebooks.build_notebook
"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

INTRO = """# truckdet: LoRA position in RF-DETR, backbone against decoder, on MIO-TCD (Colab T4)

**One-time setup**
1. *Runtime → Change runtime type →* **T4 GPU**.
2. On GitHub, create a *fine-grained personal access token* with **read-only "Contents"** access to `jin-yu-yang/truckdet`.
3. In Colab's left sidebar, open **🔑 Secrets**, add `GITHUB_TOKEN` with that token, and switch on *Notebook access*.

**How to use it.** The settings cell has one switch, `STAGE`. Set it, then *Run all*. Each stage ends with its check from the design (`docs/design.md`, §9) and tells you what to do next. Do not start a stage before the check of the stage before it passes.

| `STAGE` | Design stages | What runs | Time on a T4 |
|---|---|---|---|
| `"spike"` | 0 | Model facts on the real checkpoint; images per second and peak memory of each arm | 5 min |
| `"data"` | 1 | Download MIO-TCD (once, to Drive), convert, scene groups, split files, check sheets | 30 min the first time |
| `"checks"` | 2, 3, 4 | The COCO checkpoint with no training on val; the training loop on 16 images | 15 min |
| `"gate"` | 5 | Learning-rate runs of H and F (6 runs), the choice, **the gate** | about 8 h |
| `"main"` | 6 | Learning-rate runs of D and B (6 runs); seeds 1 and 2 of all arms (8 runs) | about 20 h |
| `"report"` | 7, 8 | The test split, one time for each of the 12 final checkpoints; the report | 1 h |

The times of `gate` and `main` are the design's estimate (an F run or a B run 2 to 3 hours, an H run or a D run about half). Stage `spike` measures the real speed and prints the hours for one run.

**If Colab disconnects or the session ends,** *Run all* again with the same `STAGE`, also in a later session: the dataset and the caches are on Drive, finished runs train nothing, and the interrupted run resumes from its last finished epoch.

**Three checks need your eyes** (stage `data`): the label completeness sheets, the scene-check sheets, and the counts. The cell prints what to count and the limit. The scene groups come from an automatic search (design §4.3); two automatic checks run before the scene check. A later stage does not start on a split that did not pass them.

**Output:** `MyDrive/truckdet/results_bundle.zip` (stage `report`), and `MyDrive/truckdet/splits/` (stage `data`; commit these files to the repository as `splits/`).
"""

SETTINGS = """STAGE = "spike"  # "spike" -> "data" -> "checks" -> "gate" -> "main" -> "report"; then Run all

REPO = "jin-yu-yang/truckdet"
BRANCH = "main"
DRIVE_ROOT = "/content/drive/MyDrive/truckdet"
CODE_DIR = "/content/truckdet"
LOCAL = "/content/data"                 # the extracted dataset: local disk, not Drive (110,000 small files)
RUNS = f"{DRIVE_ROOT}/runs"             # checkpoints, saved predictions, selection.json, gate.json
SPLITS = f"{DRIVE_ROOT}/splits"         # the split files (stage "data" writes them; commit them to the repo)
CACHE = f"{DRIVE_ROOT}/cache"           # annotations.json, embeddings.npy
CHECKS = f"{DRIVE_ROOT}/checks"         # sheets to look at; they hold dataset images, so they stay off GitHub
RESULTS = f"{DRIVE_ROOT}/results"

# Effective batch = BATCH_SIZE x ACCUM_STEPS = 16 images (design §6). Stage "spike" prints the peak memory of
# each arm at BATCH_SIZE. If arm F or B does not fit in the T4's 15 GB, use 2 x 8.
BATCH_SIZE, ACCUM_STEPS = 4, 4
assert BATCH_SIZE * ACCUM_STEPS == 16, "the design fixes the effective batch at 16 images"

# The embedding of the scene groups (stage "data"). "grid" is the second embedding of design §4.3: use it one
# time, and only if "mean" gives no limit or its split fails the scene check.
EMBEDDING = "mean"
assert EMBEDDING in ("mean", "grid")

SEEDS = [0, 1, 2]
STAGES = ["spike", "data", "checks", "gate", "main", "report"]
assert STAGE in STAGES, f"STAGE must be one of {STAGES}"


def at(stage):
    \"\"\"True when the cell of `stage` must run: it is the current stage.\"\"\"
    return STAGE == stage


def after(stage):
    \"\"\"True when the current stage comes after `stage` (its outputs must exist).\"\"\"
    return STAGES.index(STAGE) > STAGES.index(stage)


print(f"stage: {STAGE}")
"""

HELPERS = """import json
import os
import signal
import subprocess

import torch


def run(cmd: str, allowed=(0,)) -> int:
    \"\"\"Run a shell command, stream its output, and stop 'Run all' if it fails.

    The command runs in its own process group so that interrupting this cell also
    stops it (otherwise a training run would keep holding the GPU). `allowed`: exit
    codes that are not a failure (some scripts report a check's outcome that way).\"\"\"
    print(f"$ {cmd}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # stream child prints immediately
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, start_new_session=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
        returncode = proc.wait()
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    if returncode not in allowed:
        raise RuntimeError(f"command failed with exit code {returncode}: {cmd}")
    return returncode


def last_json(cmd: str) -> dict:
    \"\"\"Run a command that prints JSON on its last line, and return that JSON.\"\"\"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\\n{result.stdout}\\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → T4 GPU, then Run all again."
print("GPU:", torch.cuda.get_device_name(0))
"""

MOUNT = """from google.colab import drive

drive.mount("/content/drive")
for folder in (DRIVE_ROOT, RUNS, CACHE, CHECKS, RESULTS):
    os.makedirs(folder, exist_ok=True)
"""

CLONE = """from google.colab import userdata

try:
    token = userdata.get("GITHUB_TOKEN")
except Exception as e:
    raise RuntimeError("Add the GITHUB_TOKEN secret (see the first cell) and enable notebook access.") from e

auth_url = f"https://x-access-token:{token}@github.com/{REPO}.git"
if os.path.isdir(CODE_DIR):
    cmd = ["git", "-C", CODE_DIR, "pull", "--ff-only", auth_url, BRANCH]
else:
    cmd = ["git", "clone", "--branch", BRANCH, auth_url, CODE_DIR]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError("git failed: " + result.stderr.replace(token, "***"))
# Don't leave the token in .git/config.
subprocess.run(["git", "-C", CODE_DIR, "remote", "set-url", "origin", f"https://github.com/{REPO}.git"], check=True)
print(subprocess.run(["git", "-C", CODE_DIR, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)

run(f"pip install -q -e {CODE_DIR}")
os.chdir(CODE_DIR)
"""

SPIKE = """# Stage 0. The checks need no data. Confirm or correct the numbers of design §5.1, then read the table:
# images per second and peak memory of one training step of each arm, and the hours for one run.
if at("spike"):
    run(f"python scripts/spike.py --time --batch-size {BATCH_SIZE} --steps 20")
    raise SystemExit(
        "Stage 0 finished. If the peak memory of an arm is near 15 GB, set BATCH_SIZE, ACCUM_STEPS = 2, 8. "
        "Total training time = the sum over 20 runs of the hours in the table (12 learning-rate runs and 8 runs "
        "for seeds 1 and 2). If that is too much, design §10 gives the rule. Then set STAGE = \\"data\\" and Run all."
    )
"""

DATA = """# The dataset (CC BY-NC-SA 4.0) is downloaded once to Drive and extracted to local disk in every session.
import glob
import shutil

URL = "https://tcd.miovision.com/static/dataset"
TAR = f"{DRIVE_ROOT}/MIO-TCD-Localization.tar"   # the copy on Drive, for later sessions
LOCAL_TAR = "/content/MIO-TCD-Localization.tar"
if not glob.glob(f"{LOCAL}/**/gt_train.csv", recursive=True):
    if not os.path.exists(TAR):
        # Download to local disk first (a download straight to Drive is slow and cannot continue), then copy.
        run(f"wget -c --progress=dot:giga -O {LOCAL_TAR} {URL}/MIO-TCD-Localization.tar")
        run(f"cp {LOCAL_TAR} {TAR}.part && mv {TAR}.part {TAR}")
    source = LOCAL_TAR if os.path.exists(LOCAL_TAR) else TAR
    run(f"mkdir -p {LOCAL} && tar -xf {source} -C {LOCAL}")
DATA_ROOT = os.path.dirname(glob.glob(f"{LOCAL}/**/gt_train.csv", recursive=True)[0])
print("dataset:", DATA_ROOT)
if os.path.exists(f"{CACHE}/annotations.json") and not os.path.exists(f"{DATA_ROOT}/annotations.json"):
    for name in ("annotations.json", "counts.json"):
        shutil.copy(f"{CACHE}/{name}", f"{DATA_ROOT}/{name}")
COMMON = f"--data-root {DATA_ROOT} --splits-dir {SPLITS}"
"""

PREPARE = """# Stage 1. Converter, scene groups, split files, and the checks of design §4.5.
if at("data"):
    # The dataset's own evaluation code: confirm the classes that get an ignore region (design §4.4).
    if not glob.glob(f"{LOCAL}/code/**/localization_evaluation.py", recursive=True):
        run(f"mkdir -p {LOCAL}/code && wget -nv -O {LOCAL}/code.tar {URL}/MIO-TCD-Localization-Code.tar "
            f"&& tar -xf {LOCAL}/code.tar -C {LOCAL}/code")
    eval_code = glob.glob(f"{LOCAL}/code/**/localization_evaluation.py", recursive=True)[0]

    # Exit code 3: a count of the CSV is not the reference count. The cell goes on; find the cause before training.
    # (Boxes with zero area are dropped and printed as a note; that is not a difference.)
    counts_differ = run(f"python scripts/prepare_data.py --data-root {DATA_ROOT} --eval-code {eval_code} "
                        f"--sheets-dir {CHECKS}/labels", allowed=(0, 3))
    for name in ("annotations.json", "counts.json"):
        shutil.copy(f"{DATA_ROOT}/{name}", f"{CACHE}/{name}")

    # Scene groups: images that are linked at or above a similarity limit; an automatic search sets the limit
    # (design §4.3). Exit code 3: an automatic check failed, or the search found no limit (then there is no split).
    embeddings = f"{CACHE}/embeddings.npy" if EMBEDDING == "mean" else f"{CACHE}/embeddings_grid.npy"
    split_failed = run(f"python scripts/make_split.py --data-root {DATA_ROOT} --splits-dir {SPLITS} "
                       f"--embedding {EMBEDDING} --embeddings {embeddings} --sheets-dir {CHECKS}/scenes", allowed=(0, 3))
    run(f"python scripts/spike.py --image-dir {DATA_ROOT}/train --sheet {CHECKS}/coco_detections.jpg")

    with open(f"{SPLITS}/split_meta.json") as f:
        split_meta = json.load(f)
    if split_meta.get("limit") is None:
        if EMBEDDING == "mean":
            next_step = 'Next (design §4.3, "If no limit passes"): set EMBEDDING = "grid" and Run all, one time.'
        else:
            next_step = ("The second embedding gave no limit too. A decision is needed (design §4.3): continue with a "
                         "random image split, or stop.")
        raise SystemExit('Stage 1 stops: the search found no limit (see "[search]" and "NO LIMIT" above), so there '
                         "is no split.\\n" + next_step)
    automatic = 'FAILED (see "[check]" above). Do not go on' if split_failed else "passed"
    counts = ("DIFFERENT from the reference counts (see above): find the cause of each difference" if counts_differ
              else "equal to the reference counts")
    raise SystemExit(
        "Stage 1 finished. Before you go on:\\n"
        f"1. Counts: {counts}.\\n"
        f"2. Automatic checks (no overlap, class counts, group check, split check): {automatic}. "
        f"Limit {split_meta['limit']:.4f}; {split_meta['groups']:,} scene groups; the largest has "
        f"{split_meta['largest_group_share']:.1%} of the images.\\n"
        f"3. Label completeness: open {CHECKS}/labels. Count the vehicles and pedestrians that have no box. "
        "If more than 5% have no box, the labels are not complete: record the rate and stop.\\n"
        f"4. Scene check: open {CHECKS}/scenes. Count the scene_split pairs that show the same camera view; accept "
        "the split if 10 pairs or fewer do. Each pair is below the limit now: the question is if the limit is low "
        "enough. Do the same count on the random_split sheets, for the report. If the scene check fails, do not try "
        'other limits: set EMBEDDING = "grid" and Run all, one time (design §4.3).\\n'
        f"5. Look at {CHECKS}/scenes/largest_group.jpg: 50 images of the largest scene group. If its share is near "
        "20%, look at them before you accept the split.\\n"
        f"6. Look at {CHECKS}/coco_detections.jpg: the COCO checkpoint's detections on 20 images.\\n"
        f"7. Download {SPLITS} and commit its files to the repository as splits/ (code and split lists only; no images).\\n"
        'Then set STAGE = "checks" and Run all.'
    )
if after("data"):
    assert os.path.exists(f"{SPLITS}/split_meta.json"), f'{SPLITS} has no split files: run stage "data" first'
    with open(f"{SPLITS}/split_meta.json") as f:
        split_meta = json.load(f)
    assert split_meta.get("limit") is not None and split_meta.get("checks", {}).get("passed"), (
        f'{SPLITS} has no split that passed the automatic checks of stage "data" (an old split, or a search with no '
        'limit): run stage "data" again')
"""

CHECKS_CELL = """# Stages 2 to 4. The evaluation on real data, and the training loop on 16 images.
if at("checks"):
    # Stage 2: the COCO checkpoint with no training must score a low AP50 above zero on car and bus
    # (reference: 0.20 mAP50 for a COCO YOLOv4 on MIO-TCD). An AP near zero shows an error in the box format.
    run(f"python scripts/evaluate.py --start-model --out {RUNS}/start_model {COMMON}")
    # Stage 3 (parameter table, step-0 output, merge difference) is in spike.py and in the tests.
    # Stage 4: arm F reaches AP50 above 0.9 on 16 images; the loss of H, D and B decreases; a stopped and
    # resumed run continues the loss curve. Local disk: these runs are not results.
    shutil.rmtree("/content/overfit_runs", ignore_errors=True)
    run(f"python scripts/overfit_check.py --output-root /content/overfit_runs {COMMON}")
    raise SystemExit("Stages 2 to 4 passed. Set STAGE = \\"gate\\" and Run all.")
"""

TUNE = """def train(arm: str, seed: int, rate: str) -> None:
    \"\"\"One run and its val score. `rate`: "--lr 3e-4" (a learning-rate run) or "--selected" (the chosen rate).\"\"\"
    run(f"python scripts/train.py --config configs/arm_{arm.lower()}.yaml {rate} --seed {seed} --output-root {RUNS} "
        f"--batch-size {BATCH_SIZE} --accum-steps {ACCUM_STEPS} {COMMON}")


def tune(arm: str) -> dict:
    \"\"\"The learning-rate runs of an arm with seed 0, and the choice (design §6): the highest val AP of the final
    checkpoint; equal to 3 decimal places -> the lower rate; best rate at an end of the list -> one more step.\"\"\"
    while True:
        status = last_json(f"python scripts/select_lr.py --config configs/arm_{arm.lower()}.yaml --runs-root {RUNS} --json")
        if status["status"] == "chosen":
            print(f"arm {arm}: chosen {status}")
            return status
        if status["status"] != "run":
            raise RuntimeError(status["reason"])
        for lr in status["lr"]:
            train(arm, 0, f"--lr {lr}")
"""

GATE = """# Stage 5. Learning-rate runs of H, then of F (its heads use the rate that H chose), then the gate.
if at("gate") or after("gate"):
    tune("H")
    tune("F")
    # Exit code 3: the gate fails.
    gate_failed = run(f"python scripts/gate.py --runs-root {RUNS} {COMMON}", allowed=(0, 3))
    if gate_failed:
        raise SystemExit("The gate fails: stop here, and report that the data cannot separate the arms (design §2). "
                         "The other 14 runs are not necessary.")
    if at("gate"):
        raise SystemExit("The gate passes. Set STAGE = \\"main\\" and Run all.")
"""

MAIN = """# Stage 6. Learning-rate runs of D and B, then seeds 1 and 2 of every arm with its chosen rate.
# Seed-major order: a complete set of four arms exists after each seed.
if at("main") or after("main"):
    tune("D")
    tune("B")
    for seed in SEEDS[1:]:
        for arm in "HDBF":
            train(arm, seed, "--selected")
    if at("main"):
        raise SystemExit("All 20 runs are finished. Check above: each run has 0 skipped batches and a merge difference "
                         "below 1e-4. Then set STAGE = \\"report\\" and Run all.")
"""

REPORT = """# Stages 7 and 8. The test split is scored one time for each final checkpoint; the report reads the saved
# predictions on the CPU, and a second run gives the same text.
if at("report"):
    import pathlib
    import zipfile

    run(f"python scripts/evaluate.py --final-runs {RUNS} --split test {COMMON}")
    run(f"python scripts/analyse.py --runs-root {RUNS} --out {RESULTS} --samples 8 {COMMON}")

    bundle = pathlib.Path(DRIVE_ROOT) / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("report.md", "numbers.json", "recovered_share.png"):
            zf.write(f"{RESULTS}/{name}", f"results/{name}")
        for name in ("selection.json", "gate.json"):
            zf.write(f"{RUNS}/{name}", f"results/{name}")
        for run_dir in sorted(pathlib.Path(RUNS).iterdir()):
            for fname in ("params.json", "metrics.jsonl", "eval_val.json", "eval_test.json", "curves.png"):
                if (run_dir / fname).exists():
                    zf.write(run_dir / fname, f"results/runs/{run_dir.name}/{fname}")
        for path in sorted(pathlib.Path(SPLITS).iterdir()):
            zf.write(path, f"splits/{path.name}")
    print(f"Bundle: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB): the report, the numbers, the figure, the small "
          "files of every run and the split files. Unzip it in the repository root. The sample images are in "
          f"{RESULTS}/samples; they hold dataset images, so read the licence before you publish them.")
"""


def build_notebook():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(INTRO),
        new_code_cell(SETTINGS),
        new_code_cell(HELPERS),
        new_code_cell(MOUNT),
        new_code_cell(CLONE),
        new_markdown_cell("## Stage 0: the model"),
        new_code_cell(SPIKE),
        new_markdown_cell("## Stage 1: data and split"),
        new_code_cell(DATA),
        new_code_cell(PREPARE),
        new_markdown_cell("## Stages 2 to 4: checks before any real run"),
        new_code_cell(CHECKS_CELL),
        new_markdown_cell("## Stage 5: learning rates of H and F, and the gate"),
        new_code_cell(TUNE),
        new_code_cell(GATE),
        new_markdown_cell("## Stage 6: the main runs"),
        new_code_cell(MAIN),
        new_markdown_cell("## Stages 7 and 8: the test split and the report"),
        new_code_cell(REPORT),
    ]
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    return nb


def main() -> None:
    out = Path(__file__).resolve().parent / "colab_train.ipynb"
    nbformat.write(build_notebook(), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
