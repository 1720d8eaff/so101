import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {number}: expected an object")
            records.append(record)
    return records


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path, records):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def unique_ids(records, field="episode_id"):
    seen = set()
    for row in records:
        identifier = row.get(field)
        if not isinstance(identifier, (int, str)) or isinstance(identifier, bool):
            raise ValueError(f"Missing or invalid {field}")
        identifier = str(identifier)
        if identifier in seen:
            raise ValueError(f"Duplicate {field}: {identifier}")
        seen.add(identifier)
    return seen
