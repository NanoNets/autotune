"""progress.py's staircase: which rows can be the baseline or the best."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runs.progress import load, mark_records  # noqa: E402

HEADER = "experiment_number\ttoken/s\tsamples/s\tmemory\tstatus\thypothesis\tconclusion\n"


def _rows(tmp_path, lines):
    p = tmp_path / "results.tsv"
    p.write_text(HEADER + "".join("\t".join(map(str, l)) + "\n" for l in lines))
    return load(str(p))


def test_an_invalid_row_is_never_the_best(tmp_path):
    rows = _rows(tmp_path, [(1, 100, 10, "10GB", "ok", "baseline", "kept"),
                            (2, 900, 90, "10GB", "invalid", "x", "discarded"),
                            (3, 150, 15, "10GB", "ok", "y", "kept")])
    baseline, records = mark_records(rows)
    assert baseline == 10 and [r["n"] for r in records] == [1, 3]
    assert rows[-1]["best_so_far"] == 15                     # was 90: "9.0x" off the invalid row


def test_unreliable_and_superseded_rows_are_excluded(tmp_path):
    rows = _rows(tmp_path, [(1, 100, 10, "10GB", "ok", "baseline", "kept"),
                            (2, 300, 30, "10GB", "ok", "x", "UNRELIABLE: mfu > 1"),
                            (3, 200, 20, "10GB", "ok", "y", "kept"),
                            (4, 120, 12, "10GB", "ok", "z", "kept")])
    baseline, records = mark_records(rows, basis=4)
    assert [r["n"] for r in records] == [1, 4]               # 3 predates records_from; 2 rejected


def test_a_failed_baseline_falls_through_to_the_first_ok_row(tmp_path):
    rows = _rows(tmp_path, [(1, "", "", "", "oom", "baseline", ""),
                            (2, 100, 10, "10GB", "ok", "x", "kept")])
    assert mark_records(rows)[0] == 10
