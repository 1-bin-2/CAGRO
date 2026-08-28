import collections
import hashlib
import json
import re
import sys
from pathlib import Path


path = Path(sys.argv[1])
blob = path.read_bytes()
rows = json.loads(blob)
choice_types = {"multiple choice", "emer_ov_mc"}
option_re = re.compile(r"^\s*([A-Z])\s*[.\):：]\s*(.+?)\s*$", re.S)
answer_re = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)

semantic_keys = collections.Counter(
    (str(row.get("path", "")), str(row.get("problem", "")), str(row.get("answer", "")))
    for row in rows
)

blank_problem = sum(not str(row.get("problem", "")).strip() for row in rows)
duplicate_rows = sum(count - 1 for count in semantic_keys.values() if count > 1)
malformed_choice_rows = 0
answer_out_of_range = 0
solution_answer_mismatch = 0
missing_media = 0

for row in rows:
    media = Path(str(row.get("path", "")))
    if not media.exists():
        missing_media += 1

    expected = str(row.get("answer", "")).strip()
    match = answer_re.search(str(row.get("solution", "")))
    if not match or match.group(1).strip() != expected:
        solution_answer_mismatch += 1

    if row.get("problem_type") not in choice_types:
        continue
    options = row.get("options")
    if not isinstance(options, list) or not options:
        malformed_choice_rows += 1
        continue
    matches = [option_re.match(str(option)) for option in options]
    expected_labels = [chr(ord("A") + index) for index in range(len(options))]
    labels = [match.group(1) if match else None for match in matches]
    if not all(matches) or labels != expected_labels:
        malformed_choice_rows += 1
    if re.fullmatch(r"[A-Z]", expected):
        answer_index = ord(expected) - ord("A")
        if answer_index < 0 or answer_index >= len(options):
            answer_out_of_range += 1

report = {
    "path": str(path),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "rows": len(rows),
    "blank_problem": blank_problem,
    "duplicate_rows": duplicate_rows,
    "malformed_choice_rows": malformed_choice_rows,
    "answer_out_of_range": answer_out_of_range,
    "solution_answer_mismatch": solution_answer_mismatch,
    "missing_media": missing_media,
}
print(json.dumps(report, ensure_ascii=False, indent=2))

expected_report = {
    "sha256": "beafcc171b0facf887ba3d3f7042acb2fa91ce931d08c8f8b5f8a4aac7b66957",
    "rows": 12842,
    "blank_problem": 0,
    "duplicate_rows": 0,
    "malformed_choice_rows": 0,
    "answer_out_of_range": 0,
    "solution_answer_mismatch": 0,
    "missing_media": 0,
}
for key, expected_value in expected_report.items():
    if report[key] != expected_value:
        raise SystemExit(f"VALIDATION_FAILED: {key} expected {expected_value!r}, observed {report[key]!r}")
print("VALIDATION_OK")
