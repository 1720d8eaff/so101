"""Aggregate manually annotated trials without inventing success labels."""
import argparse
from collections import Counter, defaultdict
from .io import read_jsonl, write_json

from .recovery import FAILURE_CLASSES

FAILURES = set(FAILURE_CLASSES[1:])


def summarize(rows):
    if not rows:
        raise ValueError("Evaluation file is empty")
    seen, groups = set(), defaultdict(list)
    for row in rows:
        for field in ("policy", "checkpoint", "task", "condition", "seed", "trial_id"):
            if field not in row or row[field] is None or row[field] == "":
                raise ValueError(f"Missing {field}")
        if not isinstance(row.get("success"), bool):
            raise ValueError("success must be a manually verified boolean")
        failure = row.get("failure_type")
        if row["success"] and failure is not None:
            raise ValueError("Successful trial cannot have a failure_type")
        if not row["success"] and failure not in FAILURES:
            raise ValueError(f"Failed trial must have a failure_type from {sorted(FAILURES)}")
        key = tuple(str(row[k]) for k in ("policy", "checkpoint", "task", "condition", "seed"))
        identity = (*key, str(row["trial_id"]))
        if identity in seen:
            raise ValueError(f"Duplicate trial: {identity}")
        seen.add(identity)
        groups[key].append(row)
    result = []
    for key, group in sorted(groups.items()):
        successes = sum(x["success"] for x in group)
        result.append(dict(zip(("policy", "checkpoint", "task", "condition", "seed"), key),
                           trials=len(group), successes=successes, success_rate=successes / len(group),
                           failures=dict(Counter(x["failure_type"] for x in group if not x["success"]))))
    return {"groups": result, "note": "Results are grouped by policy/checkpoint/task/condition/training seed; no cross-seed pooling."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        write_json(args.output, summarize(read_jsonl(args.input)))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
