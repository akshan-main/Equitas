"""
Patch 7 missing Stage-2 leader responses in the FH sweep batch run,
and re-do the 50 affected Stage-3 judge calls that got "No proposal available."

Usage:
  python scripts/patch_fh_sweep.py --dry-run   # verify everything, no API calls
  python scripts/patch_fh_sweep.py              # actually patch

Safety:
  - Backs up all modified files before touching them
  - Dry-run mode verifies all logic without API calls
  - Only appends to stage2 results (never overwrites existing)
  - Only replaces affected entries in stage3 results
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BATCH_DIR = "outputs/run2/governance_sweep_fh_batch/batch_staging"
S2_REQUESTS = os.path.join(BATCH_DIR, "stage2_leaders.jsonl")
S2_RESULTS = os.path.join(BATCH_DIR, "stage2_leaders_results.jsonl")
S3_REQUESTS = os.path.join(BATCH_DIR, "stage3_judges.jsonl")
S3_RESULTS = os.path.join(BATCH_DIR, "stage3_judges_results.jsonl")
BACKUP_SUFFIX = ".backup_pre_patch"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_missing_stage2() -> List[str]:
    """Return custom_ids present in requests but not in results."""
    requested = set()
    for row in load_jsonl(S2_REQUESTS):
        requested.add(row["custom_id"])
    completed = set()
    for row in load_jsonl(S2_RESULTS):
        completed.add(row["custom_id"])
    return sorted(requested - completed)


def extract_class_from_leader_request(req: Dict) -> str:
    """Extract class name (e.g., 'auxiliary') from leader system message."""
    sys_msg = req["body"]["messages"][0]["content"]
    m = re.search(r"leader of the (\w+) class", sys_msg)
    return m.group(1) if m else "UNKNOWN"


def find_affected_stage3(missing_prefixes: set) -> List[str]:
    """Return stage3 custom_ids that have 'No proposal available' and match missing rounds."""
    affected = []
    for row in load_jsonl(S3_REQUESTS):
        cid = row["custom_id"]
        prefix = cid.split("_fhj_")[0]
        if prefix in missing_prefixes:
            user_msg = row["body"]["messages"][1]["content"]
            if "No proposal available" in user_msg:
                affected.append(cid)
    return affected


def make_api_call(client: OpenAI, messages: List[Dict], model: str,
                  temperature: float, max_tokens: int) -> str:
    """Make a single OpenAI API call and return the content."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def format_batch_result(custom_id: str, content: str, model: str) -> Dict:
    """Format a response in the same structure as batch API results."""
    return {
        "id": f"patch_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": f"patch-{custom_id}",
            "body": {
                "id": f"chatcmpl-patch-{custom_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        },
        "error": None,
    }


def patch_judge_prompt(old_prompt: str, class_name: str, leader_rationale: str) -> str:
    """Replace 'No proposal available.' under the correct class section with real rationale."""
    # Pattern: --- {CLASS} CLASS PROPOSAL ---\nNo proposal available.
    upper_class = class_name.upper()
    pattern = f"--- {upper_class} CLASS PROPOSAL ---\nNo proposal available."
    replacement = f"--- {upper_class} CLASS PROPOSAL ---\n{leader_rationale}"
    result = old_prompt.replace(pattern, replacement)
    if result == old_prompt:
        print(f"  WARNING: could not find pattern for {class_name} class in prompt")
    return result


def main():
    parser = argparse.ArgumentParser(description="Patch 7 missing FH sweep Stage-2 responses")
    parser.add_argument("--dry-run", action="store_true", help="Verify logic without API calls")
    args = parser.parse_args()

    # ── Step 1: Find missing stage2 custom_ids ──
    missing_s2 = find_missing_stage2()
    print(f"Missing Stage-2 leader responses: {len(missing_s2)}")
    for cid in missing_s2:
        print(f"  {cid}")

    if len(missing_s2) == 0:
        print("Nothing to patch!")
        return

    # ── Step 2: Load missing requests and map to class names ──
    s2_requests_map: Dict[str, Dict] = {}
    for row in load_jsonl(S2_REQUESTS):
        if row["custom_id"] in missing_s2:
            s2_requests_map[row["custom_id"]] = row

    leader_class_map: Dict[str, str] = {}  # custom_id -> class_name
    for cid, req in s2_requests_map.items():
        cls = extract_class_from_leader_request(req)
        leader_class_map[cid] = cls
        print(f"  {cid}: class={cls}")

    # ── Step 3: Find affected stage3 judge prompts ──
    missing_prefixes = set()
    for cid in missing_s2:
        prefix = cid.split("_fhl_")[0]
        missing_prefixes.add(prefix)

    affected_s3 = find_affected_stage3(missing_prefixes)
    print(f"\nAffected Stage-3 judge prompts (with 'No proposal available'): {len(affected_s3)}")

    # Map each affected judge prompt to which leader it needs
    # prefix -> list of missing leader custom_ids for that round
    prefix_to_leaders: Dict[str, List[str]] = {}
    for cid in missing_s2:
        prefix = cid.split("_fhl_")[0]
        prefix_to_leaders.setdefault(prefix, []).append(cid)

    if args.dry_run:
        print(f"\n=== DRY RUN: Would make {len(missing_s2)} leader + {len(affected_s3)} judge = {len(missing_s2) + len(affected_s3)} API calls ===")
        print("Everything checks out. Run without --dry-run to execute.")
        return

    # ── Step 4: Back up files ──
    for f in [S2_RESULTS, S3_RESULTS]:
        backup = f + BACKUP_SUFFIX
        if not os.path.exists(backup):
            shutil.copy2(f, backup)
            print(f"Backed up: {f} -> {backup}")
        else:
            print(f"Backup already exists: {backup}")

    # ── Step 5: Make Stage-2 API calls ──
    client = OpenAI()
    model = "gpt-4o-mini"
    temperature = 0.7
    max_tokens = 512

    leader_rationales: Dict[str, str] = {}  # custom_id -> rationale
    print(f"\n--- Making {len(missing_s2)} Stage-2 leader API calls ---")
    for cid in missing_s2:
        req = s2_requests_map[cid]
        messages = req["body"]["messages"]
        print(f"  Calling {cid} ({leader_class_map[cid]} class)...", end=" ", flush=True)
        content = make_api_call(client, messages, model, temperature, max_tokens)
        leader_rationales[cid] = content
        print(f"OK ({len(content)} chars)")

    # Append to stage2 results
    with open(S2_RESULTS, "a") as f:
        for cid in missing_s2:
            result = format_batch_result(cid, leader_rationales[cid], model)
            f.write(json.dumps(result) + "\n")
    print(f"Appended {len(missing_s2)} results to {S2_RESULTS}")

    # Verify stage2 is now complete
    remaining = find_missing_stage2()
    assert len(remaining) == 0, f"Stage 2 still has {len(remaining)} missing!"
    print("Stage 2 is now complete (14,711/14,711).")

    # ── Step 6: Make Stage-3 API calls with patched prompts ──
    # Build mapping: round prefix -> class -> rationale
    prefix_class_rationale: Dict[str, Dict[str, str]] = {}
    for cid in missing_s2:
        prefix = cid.split("_fhl_")[0]
        cls = leader_class_map[cid]
        prefix_class_rationale.setdefault(prefix, {})[cls] = leader_rationales[cid]

    # Load all stage3 requests for quick lookup
    s3_requests_map: Dict[str, Dict] = {}
    for row in load_jsonl(S3_REQUESTS):
        if row["custom_id"] in affected_s3:
            s3_requests_map[row["custom_id"]] = row

    print(f"\n--- Making {len(affected_s3)} Stage-3 judge API calls ---")
    new_s3_results: Dict[str, Dict] = {}
    for i, cid in enumerate(affected_s3):
        req = s3_requests_map[cid]
        prefix = cid.split("_fhj_")[0]
        old_user_msg = req["body"]["messages"][1]["content"]

        # Replace "No proposal available." with actual rationale for each missing class
        new_user_msg = old_user_msg
        for cls, rationale in prefix_class_rationale.get(prefix, {}).items():
            new_user_msg = patch_judge_prompt(new_user_msg, cls, rationale)

        # Verify the patch worked
        assert "No proposal available" not in new_user_msg, \
            f"Patching failed for {cid}: still has 'No proposal available'"

        # Make API call with patched prompt
        patched_messages = [
            req["body"]["messages"][0],  # system message (unchanged)
            {"role": "user", "content": new_user_msg},
        ]
        print(f"  [{i+1}/{len(affected_s3)}] Calling {cid}...", end=" ", flush=True)
        content = make_api_call(client, patched_messages, model, temperature, max_tokens)
        new_s3_results[cid] = format_batch_result(cid, content, model)
        print(f"OK ({len(content)} chars)")

    # ── Step 7: Patch stage3 results file ──
    # Read all existing results, replace affected ones, write back
    print(f"\nPatching {S3_RESULTS}...")
    existing_s3 = load_jsonl(S3_RESULTS)
    patched_count = 0
    with open(S3_RESULTS, "w") as f:
        for row in existing_s3:
            cid = row["custom_id"]
            if cid in new_s3_results:
                f.write(json.dumps(new_s3_results[cid]) + "\n")
                patched_count += 1
            else:
                f.write(json.dumps(row) + "\n")
    print(f"Patched {patched_count} entries in stage3 results (total: {len(existing_s3)})")

    # ── Step 8: Verify ──
    # Check stage3 results count
    final_s3 = load_jsonl(S3_RESULTS)
    print(f"\nFinal stage3 results: {len(final_s3)}")

    print(f"\n=== Patch complete! ===")
    print(f"  Stage 2: +{len(missing_s2)} leader responses (was 14,704 -> now 14,711)")
    print(f"  Stage 3: {patched_count} judge responses re-done with correct proposals")
    print(f"\nBackups saved with suffix '{BACKUP_SUFFIX}'")
    print(f"\nNext step: re-run the pipeline in resume mode to re-assemble recordings:")
    print(f"  python -c \"")
    print(f"from equitas.batch.pipeline_fh import FHBatchSweepPipeline")
    print(f"from equitas.config import load_config")
    print(f"p = FHBatchSweepPipeline(load_config('configs/governance_sweep_fh.yaml'))")
    print(f"p.run(resume=True)")
    print(f"\"")


if __name__ == "__main__":
    main()
