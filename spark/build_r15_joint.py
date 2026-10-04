"""Rebuild round 15's joint training file (documents-v1 train + hard-v1 train + devtools-v1 train) and check it against
the sha256 recorded on the research-archive-2026-09-24 tag (evals/round15/joint/manifest.json)."""
import hashlib
from pathlib import Path
from kev.suite import load_split, write_jsonl

JOINT_SHA = "1bf25e29582c11a33de334c4c3847f7cc5fe8b318ad05c6ad17162de34205950"
SKILLS_SHA = "14db86b947721b878f314e6ceafc858082faafe7cf17fcb11d2869f725884fa2"
parts = {name: load_split(f"evals/{name}", "train") for name in ("documents-v1", "hard-v1", "devtools-v1")}
for name, rows in parts.items(): print(name, len(rows))
out = Path("evals/round15/joint"); out.mkdir(parents=True, exist_ok=True)
skills = parts["hard-v1"] + parts["devtools-v1"]
write_jsonl(out / "skills.jsonl", skills)
print("skills sha ok:", hashlib.sha256((out / "skills.jsonl").read_bytes()).hexdigest() == SKILLS_SHA)
write_jsonl(out / "train.jsonl", parts["documents-v1"] + skills)
digest = hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
print("joint records", len(parts["documents-v1"]) + len(skills), "sha ok:", digest == JOINT_SHA, digest)
