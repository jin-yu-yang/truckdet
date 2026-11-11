import ast
import glob
import os
import re
from pathlib import Path

import nbformat

from notebooks.build_notebook import build_notebook

ROOT = Path(__file__).parent.parent


def code_cells(nb) -> list[str]:
    return [c.source for c in nb.cells if c.cell_type == "code"]


def test_notebook_file_is_up_to_date():
    """Edit notebooks/build_notebook.py and re-run it; the .ipynb is generated."""
    on_disk = nbformat.read(ROOT / "notebooks" / "colab_train.ipynb", as_version=4)
    assert code_cells(on_disk) == code_cells(build_notebook())
    assert [c.source for c in on_disk.cells if c.cell_type == "markdown"] == \
        [c.source for c in build_notebook().cells if c.cell_type == "markdown"]


def test_notebook_is_valid_and_every_cell_parses():
    nb = build_notebook()
    nbformat.validate(nb)
    for source in code_cells(nb):
        ast.parse(source)
    assert nb.metadata["colab"]["gpuType"] == "T4"


def test_commands_use_scripts_configs_and_options_that_exist():
    source = "\n".join(code_cells(build_notebook()))
    scripts = set(re.findall(r"python (scripts/\w+\.py)", source))
    assert scripts == {f"scripts/{n}.py" for n in ("spike", "prepare_data", "make_split", "evaluate", "overfit_check",
                                                   "train", "select_lr", "gate", "analyse")}
    for path in scripts:
        text = (ROOT / path).read_text()
        for line in source.splitlines():
            if f"python {path}" in line:
                for option in re.findall(r"(--[a-z][a-z-]+)", line):
                    assert f'"{option}"' in text, f"{path} has no option {option}"
    assert all((ROOT / "configs" / f"arm_{arm}.yaml").exists() for arm in "hdbf")
    assert 'f"python scripts/train.py --config configs/arm_{arm.lower()}.yaml' in source


def test_stages_run_in_the_order_of_the_design():
    source = "\n".join(code_cells(build_notebook()))
    assert 'STAGES = ["spike", "data", "checks", "gate", "main", "report"]' in source
    order = [source.index(marker) for marker in ('if at("spike")', 'if at("data")', 'if at("checks")', 'tune("H")', 'tune("F")',
                                                 "scripts/gate.py", 'tune("D")', 'tune("B")', "--split test", "scripts/analyse.py")]
    assert order == sorted(order)
    assert "BATCH_SIZE * ACCUM_STEPS == 16" in source  # the effective batch of the design
    assert 'for arm in "HDBF"' in source and "SEEDS[1:]" in source  # seeds 1 and 2 of every arm
    assert "The gate fails: stop here" in source


def test_dataset_folder_is_the_one_with_the_images(tmp_path):
    """Stage `data`, fourth run: the dataset's code archive has a `gt_train.csv` too. In a session that had the
    archive from an earlier run, the cell took it for the dataset, and the label sheets found no image."""
    cell = next(source for source in code_cells(build_notebook()) if "def dataset_roots" in source)
    functions = [node for node in ast.parse(cell).body if isinstance(node, ast.FunctionDef)]
    assert [f.name for f in functions] == ["is_dataset", "dataset_roots"]
    namespace = {"os": os, "glob": glob, "LOCAL": str(tmp_path)}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "data cell", "exec"), namespace)
    dataset_roots, is_dataset = namespace["dataset_roots"], namespace["is_dataset"]

    rows = "".join(f"{k:08d},car,1,2,30,40\n" for k in (0, 0, 1, 2))  # 3 images, one with two boxes
    code = tmp_path / "code" / "A-Code"  # the code archive: the CSV, and a train folder with one example image
    dataset = tmp_path / "MIO-TCD-Localization"
    for folder in (code, dataset):
        (folder / "train").mkdir(parents=True)
        (folder / "gt_train.csv").write_text(rows)
    (code / "train" / "00000000.jpg").write_bytes(b"")
    assert dataset_roots() == []  # no folder has its images: the cell extracts the archive
    for k in (0, 1):
        (dataset / "train" / f"{k:08d}.jpg").write_bytes(b"")
    assert dataset_roots() == []  # an extraction that was stopped in the middle is not the dataset
    (dataset / "train" / "00000002.jpg").write_bytes(b"")
    assert dataset_roots() == [str(dataset)] and not is_dataset(str(code))
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "gt_train.csv").write_text(rows)  # a CSV with no train folder
    assert dataset_roots() == [str(dataset)]
    assert "DATA_ROOT = roots[0]" in cell and "assert roots" in cell
