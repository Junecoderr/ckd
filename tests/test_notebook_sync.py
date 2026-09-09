"""Guard against the teaching notebook drifting away from src/.

`notebooks/CKD_Classification_Colab.ipynb` deliberately restates the pipeline
cell by cell instead of importing it -- that repetition is the point of a
teaching notebook, and collapsing it into `from ckd_pipeline import *` would
leave a reader with nothing to read.

The cost of that choice is drift: the notebook and the script can silently
disagree. These tests pin down the parts where a disagreement would actually
change a result -- the column groups, the random seed, the fold count, and the
target encoding -- and fail loudly if the two copies diverge.

Run with:  PYTHONPATH=src pytest -q
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ckd_pipeline import CATEGORICAL, NUMERIC, N_SPLITS, RANDOM_STATE

NOTEBOOK = ROOT / "notebooks" / "CKD_Classification_Colab.ipynb"


@pytest.fixture(scope="module")
def notebook_source():
    if not NOTEBOOK.exists():
        pytest.skip("teaching notebook not present")
    nb = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(c["source"])
                     for c in nb["cells"] if c["cell_type"] == "code")


def _literal(source, name):
    """Value of the last top-level `name = <literal>` assignment."""
    pattern = rf"^{name}\s*=\s*(.+?)$"
    matches = re.findall(pattern, source, flags=re.MULTILINE)
    if not matches:
        pytest.fail(f"notebook never assigns {name}")
    return ast.literal_eval(matches[-1].strip())


def test_notebook_is_valid_json_and_has_cells():
    nb = json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"]) > 0
    assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_numeric_columns_match(notebook_source):
    assert _literal(notebook_source, "NUMERIC") == NUMERIC


def test_categorical_columns_match(notebook_source):
    assert _literal(notebook_source, "CATEGORICAL") == CATEGORICAL


def test_seed_and_fold_count_match(notebook_source):
    assert _literal(notebook_source, "RANDOM_STATE") == RANDOM_STATE
    assert _literal(notebook_source, "N_SPLITS") == N_SPLITS


def test_target_encoding_matches(notebook_source):
    """CKD must be the POSITIVE class in both copies, or Sensitivity flips."""
    assert '"ckd": 1' in notebook_source or "'ckd': 1" in notebook_source
    assert '"notckd": 0' in notebook_source or "'notckd': 0" in notebook_source


def test_notebook_outputs_are_stripped():
    """Committed notebooks with embedded outputs make diffs unreadable and can
    carry stale numbers that contradict results/."""
    nb = json.loads(NOTEBOOK.read_text())
    heavy = [i for i, c in enumerate(nb["cells"])
             if c.get("outputs") or c.get("execution_count") is not None]
    assert not heavy, (
        f"cells {heavy} still carry execution output; "
        "clear outputs before committing")
