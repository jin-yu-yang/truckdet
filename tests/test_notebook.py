import ast
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
