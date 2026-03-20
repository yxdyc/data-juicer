import json

from data_juicer.utils.constant import Fields
from data_juicer.utils.jsonl_lenient_loader import (
    dataset_from_lenient_jsonl_files,
    iter_lenient_jsonl_records,
)


def test_iter_lenient_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    huge_int = 2**65  # ujson may reject; stdlib ok
    lines = [
        json.dumps({"ok": 1, "id": huge_int}),
        "not json",
        json.dumps(["not", "an", "object"]),
        json.dumps({"ok": 2}),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")

    rows = list(
        iter_lenient_jsonl_records(
            [(str(p), ".jsonl")],
            add_suffix_column=False,
        )
    )
    assert len(rows) == 2
    assert rows[0]["ok"] == 1
    assert rows[0]["id"] == huge_int
    assert rows[1]["ok"] == 2


def test_iter_lenient_jsonl_adds_suffix(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
    rows = list(
        iter_lenient_jsonl_records(
            [(str(p), ".jsonl")],
            add_suffix_column=True,
        )
    )
    assert len(rows) == 1
    assert rows[0][Fields.suffix] == ".jsonl"


def test_dataset_from_lenient_jsonl_files(tmp_path):
    p = tmp_path / "t.jsonl"
    body = (
        json.dumps({"a": 1}) + "\n" + "oops\n" + json.dumps({"a": 2}) + "\n"
    )
    p.write_text(body, encoding="utf-8")
    ds = dataset_from_lenient_jsonl_files(
        [(str(p), ".jsonl")],
        add_suffix_column=False,
    )
    assert len(ds) == 2
    assert list(ds["a"]) == [1, 2]
