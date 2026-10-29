"""Join frozen total-match decisions to receipt reference item counts.

This is a descriptive, CPU-only posthoc analysis. It verifies the source hashes
recorded in analysis_evidence.json and requires one unambiguous reference join
per selected receipt before writing any outputs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
import os
PROJECT_ROOT = os.environ.get('RECEIPT_PROJECT_ROOT', '.')  # folder that holds the dated project directories


OUT = Path(__file__).resolve().parent
TASK = Path(PROJECT_ROOT, "2026-09-30", "task")  # private inputs live under the project root, not beside this script
EVIDENCE = OUT / "analysis_evidence.json"
CMP = Path(
    PROJECT_ROOT, "2026-10-05", "task", "deliverables", "qwen3_5_2b_comparison_v1_recovery",
    "receipt_photo_model_results_qwen35_2b_2026-10-06", "comparison_all_views.csv",
)


def frozen_key(path: Path) -> str:
    """Hash keys in analysis_evidence.json are recorded relative to the project root as <PROJECT_ROOT>\\..."""
    rel = Path(path).resolve().relative_to(Path(PROJECT_ROOT).resolve())
    return "<PROJECT_ROOT>\\" + "\\".join(rel.parts)


LEG = TASK / "deliverables/receipy_vscode_workbench/data/labels/legacy_reference_v3_1.json"
NEW = TASK / "deliverables/receipy_vscode_workbench/data/labels/receipy_references_v2.json"
ROW_SCORES = TASK / "photo_pilot/receipy_extension/scoring_v2/row_receipt_scores.csv"
DEST = OUT / "item_count_complexity_addendum"
ROUTES = [
    "Tesseract+rules",
    "LayoutLMv3",
    "TrOCR+LayoutLMv3",
    "Donut",
    "GLM-OCR",
    "Qwen3.5-2B",
]
CONDITIONS = ["clear", "scrunched"]
COUNT_BINS = [(1, 3, "1-3"), (4, 5, "4-5"), (6, math.inf, "6+")]
UNKNOWN_TOKENS = {"", "unknown", "unreadable", "not printed", "n/a"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def to_decimal(value: object) -> Decimal | None:
    try:
        if value is None or not str(value).strip():
            return None
        return Decimal(str(value).strip())
    except InvalidOperation:
        return None


def usable_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in UNKNOWN_TOKENS else text


def item_bin(count: int | None) -> str:
    if count is None:
        return "not binned: incomplete/unknown inventory"
    for low, high, label in COUNT_BINS:
        if low <= count <= high:
            return label
    return "not binned: zero observed product rows"


def wilson(successes: int, n: int) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0, center - radius), min(1, center + radius)


def pctl(values: list[int], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    expected_hashes = evidence["input_files"]
    inputs = [CMP, LEG, NEW, ROW_SCORES]
    verified = []
    for path in inputs:
        key = frozen_key(path)
        expected = expected_hashes.get(key)
        actual = sha256(path)
        if expected is None:
            raise AssertionError(f"No frozen input hash recorded for {path}")
        if actual != expected["sha256"]:
            raise AssertionError(f"Input hash changed: {path}")
        verified.append(
            {
                "path": key,
                "sha256": actual,
                "bytes": path.stat().st_size,
                "match_to_analysis_evidence": True,
            }
        )

    legacy_records = json.loads(LEG.read_text(encoding="utf-8"))
    new_records = json.loads(NEW.read_text(encoding="utf-8"))["records"]
    row_scores = read_csv(ROW_SCORES)
    cmp = read_csv(CMP)

    # The legacy source has no row_type field; its line_items are the listed
    # reference rows. The newer source distinguishes all raw entries, product
    # rows, and the separate row scorer's eligible row candidates.
    row_score_counts: dict[str, set[int]] = defaultdict(set)
    for row in row_scores:
        if row.get("condition") == "clear" and row.get("route") == "Donut":
            row_score_counts[row["receipt_id"]].add(int(row["gt_rows"]))
    if any(len(counts) != 1 for counts in row_score_counts.values()):
        raise AssertionError("Eligible row-candidate count disagrees across references")
    eligible_candidates = {rid: next(iter(counts)) for rid, counts in row_score_counts.items()}

    refs: dict[str, dict[str, object]] = {}
    name_rows: list[dict[str, object]] = []
    for ref in legacy_records:
        rid = ref["receipt_id"]
        items = ref.get("line_items") or []
        names = [usable_text(item.get("name")) for item in items]
        names = [name for name in names if name is not None]
        field_uncertainty = [item.get("field_uncertainty", {}) for item in items]
        unknown_name_n = sum(usable_text(item.get("name")) is None for item in items)
        unknown_quantity_n = sum(usable_text(item.get("quantity")) is None for item in items)
        unknown_amount_n = sum(usable_text(item.get("item_total")) is None for item in items)
        refs[rid] = {
            "reference_id": rid,
            "source_dataset": "legacy_reference_v3_1",
            "raw_reference_entries": len(items),
            "product_item_count": len(items),
            "product_count_definition": "all legacy line_items; schema has no row_type, each listed line item treated as a product row",
            "score_eligible_row_candidates": None,
            "count_status": "transcribed row count; full receipt inventory completeness not independently asserted",
            "complete_visible_inventory_candidate": "not encoded",
            "known_product_names": len(names),
            "unknown_name_entries": unknown_name_n,
            "unknown_quantity_entries": unknown_quantity_n,
            "unknown_amount_entries": unknown_amount_n,
            "total_reference": ref.get("total"),
            "total_reference_status": ref.get("reviewer_status", "assistant_provisional_unverified"),
            "currency": ref.get("currency", ""),
            "item_name_chars_total": sum(map(len, names)),
            "item_name_chars_mean": statistics.mean(map(len, names)) if names else None,
            "item_name_chars_median": statistics.median(map(len, names)) if names else None,
        }
        for index, item in enumerate(items, 1):
            name = usable_text(item.get("name"))
            if name is not None:
                name_rows.append(
                    {
                        "reference_id": rid,
                        "source_dataset": "legacy_reference_v3_1",
                        "row_index": index,
                        "row_type": "legacy line_item (no row_type field)",
                        "name_chars": len(name),
                        "name_uncertainty": (item.get("field_uncertainty") or {}).get("name", "not recorded"),
                        "human_verified": False,
                    }
                )

    for ref in new_records:
        rid = ref["receipt_id"]
        items = ref.get("items") or []
        product_items = [item for item in items if item.get("row_type") == "product"]
        eligible_row_candidates = eligible_candidates.get(rid)
        names: list[str] = []
        for item in product_items:
            name = usable_text(item.get("name"))
            if name is not None and item.get("name_score_eligible") is not False:
                names.append(name)
        complete = bool(ref.get("complete_visible_inventory_candidate"))
        count_status = (
            "full photographed ledger candidate; assistant-audited, human verification pending"
            if complete
            else "partial/ambiguous reference; observed rows are a lower bound and excluded from item-count bins"
        )
        refs[rid] = {
            "reference_id": rid,
            "source_dataset": "receipy_references_v2",
            "raw_reference_entries": len(items),
            "product_item_count": len(product_items),
            "product_count_definition": "items with row_type=product; excludes discounts, promotions, adjustments, and other raw entries",
            "score_eligible_row_candidates": eligible_row_candidates,
            "count_status": count_status,
            "complete_visible_inventory_candidate": complete,
            "known_product_names": len(names),
            "unknown_name_entries": sum(
                item.get("name_score_eligible") is False or usable_text(item.get("name")) is None
                for item in product_items
            ),
            "unknown_quantity_entries": sum(usable_text(item.get("quantity")) is None for item in product_items),
            "unknown_amount_entries": sum(usable_text(item.get("line_total")) is None for item in product_items),
            "total_reference": ref.get("total"),
            "total_reference_status": ref.get("human_verification", "pending"),
            "currency": ref.get("currency", ""),
            "item_name_chars_total": sum(map(len, names)),
            "item_name_chars_mean": statistics.mean(map(len, names)) if names else None,
            "item_name_chars_median": statistics.median(map(len, names)) if names else None,
        }
        for index, item in enumerate(product_items, 1):
            name = usable_text(item.get("name"))
            if name is not None and item.get("name_score_eligible") is not False:
                name_rows.append(
                    {
                        "reference_id": rid,
                        "source_dataset": "receipy_references_v2",
                        "row_index": index,
                        "row_type": item.get("row_type", ""),
                        "name_chars": len(name),
                        "name_uncertainty": item.get("confidence", "not recorded"),
                        "human_verified": False,
                    }
                )

    # Restrict to the saved primary 50-pair/four? Six-route comparison. The
    # 4B placeholder and non-primary rows are excluded exactly as in the saved
    # current score table. Reference IDs must map one-to-one by explicit prefix.
    selected = [
        row for row in cmp
        if row.get("route") in ROUTES
        and row.get("condition") in CONDITIONS
        and row.get("route_status") in {"ok", "generated"}
    ]
    def canonical_id(score_id: str) -> str:
        return score_id[len("legacy_"):] if score_id.startswith("legacy_") else score_id

    mapped_ids = {canonical_id(row["receipt_id"]) for row in selected}
    if len(mapped_ids) != 50:
        raise AssertionError(f"Expected 50 selected receipts, found {len(mapped_ids)}")
    if mapped_ids != set(refs).intersection(mapped_ids):
        missing = sorted(mapped_ids - set(refs))
        raise AssertionError(f"Receipt IDs lack a label-reference mapping: {missing}")
    if len(selected) != 50 * 2 * len(ROUTES):
        raise AssertionError(f"Expected 600 scored rows, found {len(selected)}")
    keys = [(row["receipt_id"], row["condition"], row["route"]) for row in selected]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate receipt-condition-route score keys")
    if set(mapped_ids) != set(refs.keys()) - {"new_010_clear", "new_010_scrunched"}:
        raise AssertionError("Selected 50-receipt cohort does not match the reference set")

    # All available known totals must agree numerically across the join. This
    # also validates that legacy prefix normalization and direct new_* IDs are
    # not accidentally cross-mapped.
    for row in selected:
        rid = canonical_id(row["receipt_id"])
        ref = refs[rid]
        if row.get("reference_known") == "1":
            if to_decimal(row.get("reference_total")) != to_decimal(ref["total_reference"]):
                raise AssertionError(f"Saved total label differs after ID join: {row['receipt_id']}")

    # Keep incomplete or fragment-only item references visible but do not
    # assign them a false zero-item bin. Legacy counts are annotated row counts,
    # not independently confirmed physical item counts.
    receipt_info: dict[str, dict[str, object]] = {}
    for score_id in sorted({row["receipt_id"] for row in selected}):
        rid = canonical_id(score_id)
        ref = refs[rid]
        count_usable = not (
            ref["source_dataset"] == "receipy_references_v2"
            and not ref["complete_visible_inventory_candidate"]
        )
        info = dict(ref)
        info.update(
            {
                "score_receipt_id": score_id,
                "reference_id_join_key": rid,
                "product_count_bin": item_bin(int(ref["product_item_count"]) if count_usable else None),
                "raw_entry_count_bin": item_bin(int(ref["raw_reference_entries"]) if count_usable else None),
                "item_count_usable_for_bins": count_usable,
            }
        )
        receipt_info[score_id] = info

    per_receipt_rows: list[dict[str, object]] = []
    for row in sorted(selected, key=lambda x: (x["receipt_id"], x["condition"], ROUTES.index(x["route"]))):
        info = receipt_info[row["receipt_id"]]
        known = row.get("reference_known") == "1"
        per_receipt_rows.append(
            {
                "score_receipt_id": row["receipt_id"],
                "reference_id_join_key": info["reference_id_join_key"],
                "pair_id": row["pair_id"],
                "condition": row["condition"],
                "route": row["route"],
                "route_status": row["route_status"],
                "reference_total": row.get("reference_total", ""),
                "reference_known": known,
                "reference_status": row.get("reference_status", ""),
                "saved_exact_match_flag": row.get("exact_match", ""),
                "scored_exact_match": row.get("exact_match", "") if known else "",
                "item_reference_source": info["source_dataset"],
                "raw_reference_entries": info["raw_reference_entries"],
                "product_item_count": info["product_item_count"],
                "score_eligible_row_candidates": info["score_eligible_row_candidates"],
                "product_count_bin": info["product_count_bin"],
                "raw_entry_count_bin": info["raw_entry_count_bin"],
                "item_count_usable_for_bins": info["item_count_usable_for_bins"],
                "count_status": info["count_status"],
                "complete_visible_inventory_candidate": info["complete_visible_inventory_candidate"],
                "known_product_names": info["known_product_names"],
                "unknown_name_entries": info["unknown_name_entries"],
                "unknown_quantity_entries": info["unknown_quantity_entries"],
                "unknown_amount_entries": info["unknown_amount_entries"],
                "item_name_chars_total": info["item_name_chars_total"],
                "item_name_chars_mean": info["item_name_chars_mean"],
                "item_name_chars_median": info["item_name_chars_median"],
            }
        )
    write_csv(DEST / "per_receipt_total_item_counts.csv", per_receipt_rows)

    # One row per receipt/condition/route is the source-grain result above;
    # aggregate by both product-only count and raw-entry count to expose the
    # sensitivity to non-product promotions/discounts in the newer schema.
    stratified: list[dict[str, object]] = []
    for count_basis, bin_field in [
        ("product_item_count", "product_count_bin"),
        ("raw_reference_entry_count", "raw_entry_count_bin"),
    ]:
        for route in ROUTES:
            for condition in CONDITIONS:
                for _, _, bin_name in COUNT_BINS:
                    bucket = [
                        row for row in per_receipt_rows
                        if row["route"] == route
                        and row["condition"] == condition
                        and row[bin_field] == bin_name
                    ]
                    # Every receipt has one row per condition/model; count the
                    # receipt once, while labels are condition-specific.
                    known_rows = [row for row in bucket if row["reference_known"]]
                    correct = sum(str(row["scored_exact_match"]) == "1" for row in known_rows)
                    n_known = len(known_rows)
                    unknown_n = sum(not row["reference_known"] for row in bucket)
                    low, high = wilson(correct, n_known)
                    stratified.append(
                        {
                            "count_basis": count_basis,
                            "item_count_bin": bin_name,
                            "route": route,
                            "condition": condition,
                            "receipts_in_bin_n": len({row["score_receipt_id"] for row in bucket}),
                            "scored_view_rows_n": len(bucket),
                            "known_total_label_n": n_known,
                            "unknown_total_label_n": unknown_n,
                            "exact_total_matches_n": correct,
                            "accuracy_over_known_labels": correct / n_known if n_known else None,
                            "wilson95_low": low,
                            "wilson95_high": high,
                            "bin_inventory_scope_note": "Observed transcribed reference row counts; legacy completeness not independently asserted; Receipy fragment excluded.",
                        }
                    )
    # Include excluded incomplete/unknown item-count references separately so
    # the cohort coverage is visible rather than silently lost.
    for count_basis, bin_field in [
        ("product_item_count", "product_count_bin"),
        ("raw_reference_entry_count", "raw_entry_count_bin"),
    ]:
        for route in ROUTES:
            for condition in CONDITIONS:
                bucket = [
                    row for row in per_receipt_rows
                    if row["route"] == route
                    and row["condition"] == condition
                    and row[bin_field] == "not binned: incomplete/unknown inventory"
                ]
                known_rows = [row for row in bucket if row["reference_known"]]
                correct = sum(str(row["scored_exact_match"]) == "1" for row in known_rows)
                low, high = wilson(correct, len(known_rows))
                stratified.append(
                    {
                        "count_basis": count_basis,
                        "item_count_bin": "not binned: incomplete/unknown inventory",
                        "route": route,
                        "condition": condition,
                        "receipts_in_bin_n": len({row["score_receipt_id"] for row in bucket}),
                        "scored_view_rows_n": len(bucket),
                        "known_total_label_n": len(known_rows),
                        "unknown_total_label_n": len(bucket) - len(known_rows),
                        "exact_total_matches_n": correct,
                        "accuracy_over_known_labels": correct / len(known_rows) if known_rows else None,
                        "wilson95_low": low,
                        "wilson95_high": high,
                        "bin_inventory_scope_note": "Shown separately; excluded from count strata because receipt item count is incomplete or ambiguous.",
                    }
                )
    write_csv(DEST / "total_accuracy_by_item_count_bin.csv", stratified)

    # Per-name length rows plus compact distributions. Receipt-level aggregate
    # item-name characters are a proxy for item-list text volume only, not a
    # full receipt transcription or OCR token count.
    write_csv(DEST / "product_name_length_rows.csv", name_rows)
    names_by_source: dict[str, list[int]] = defaultdict(list)
    for row in name_rows:
        names_by_source[str(row["source_dataset"])].append(int(row["name_chars"]))
    length_summary = []
    for source in ["legacy_reference_v3_1", "receipy_references_v2"]:
        values = names_by_source[source]
        if not values:
            continue
        length_summary.append(
            {
                "source_dataset": source,
                "product_name_strings_n": len(values),
                "mean_chars": statistics.mean(values),
                "median_chars": statistics.median(values),
                "p25_chars": pctl(values, 0.25),
                "p75_chars": pctl(values, 0.75),
                "min_chars": min(values),
                "max_chars": max(values),
                "note": "Known/score-eligible reference product names only; descriptive string length, not a validated difficulty measure.",
            }
        )
    all_values = [int(row["name_chars"]) for row in name_rows]
    if all_values:
        length_summary.append(
            {
                "source_dataset": "combined",
                "product_name_strings_n": len(all_values),
                "mean_chars": statistics.mean(all_values),
                "median_chars": statistics.median(all_values),
                "p25_chars": pctl(all_values, 0.25),
                "p75_chars": pctl(all_values, 0.75),
                "min_chars": min(all_values),
                "max_chars": max(all_values),
                "note": "Known/score-eligible reference product names only; descriptive string length, not a validated difficulty measure.",
            }
        )
    write_csv(DEST / "product_name_length_summary.csv", length_summary)

    receipt_length_rows = [
        info for info in receipt_info.values()
        if info["item_count_usable_for_bins"]
    ]
    receipt_length_summary = []
    for source in ["legacy_reference_v3_1", "receipy_references_v2", "combined"]:
        rows = [
            info for info in receipt_length_rows
            if source == "combined" or info["source_dataset"] == source
        ]
        values = [int(info["item_name_chars_total"]) for info in rows]
        if values:
            receipt_length_summary.append(
                {
                    "source_dataset": source,
                    "receipts_n": len(values),
                    "mean_known_product_name_chars_per_receipt": statistics.mean(values),
                    "median_known_product_name_chars_per_receipt": statistics.median(values),
                    "p25_chars": pctl(values, 0.25),
                    "p75_chars": pctl(values, 0.75),
                    "min_chars": min(values),
                    "max_chars": max(values),
                    "known_product_names_n": sum(int(info["known_product_names"]) for info in rows),
                    "note": "Sum of known item-name string lengths only; not full receipt text length or OCR token count.",
                }
            )
    write_csv(DEST / "receipt_item_name_text_volume_proxy.csv", receipt_length_summary)

    count_distribution: list[dict[str, object]] = []
    count_usable_info = [info for info in receipt_info.values() if info["item_count_usable_for_bins"]]
    for count_basis, key in [
        ("product_item_count", "product_item_count"),
        ("raw_reference_entry_count", "raw_reference_entries"),
    ]:
        for _, _, label in COUNT_BINS:
            in_bin = [info for info in count_usable_info if item_bin(int(info[key])) == label]
            count_distribution.append(
                {
                    "count_basis": count_basis,
                    "item_count_bin": label,
                    "receipts_n": len(in_bin),
                    "legacy_transcribed_count_receipts_n": sum(info["source_dataset"] == "legacy_reference_v3_1" for info in in_bin),
                    "Receipy_full_ledger_candidate_receipts_n": sum(info["source_dataset"] == "receipy_references_v2" for info in in_bin),
                    "unknown_inventory_receipts_excluded_n": sum(not info["item_count_usable_for_bins"] for info in receipt_info.values()),
                }
            )
    product_bin = {info["score_receipt_id"]: item_bin(int(info["product_item_count"])) for info in receipt_info.values() if info["item_count_usable_for_bins"]}
    raw_bin = {info["score_receipt_id"]: item_bin(int(info["raw_reference_entries"])) for info in receipt_info.values() if info["item_count_usable_for_bins"]}
    changed_bin_ids = [rid for rid in product_bin if product_bin[rid] != raw_bin[rid]]
    count_distribution.append(
        {
            "count_basis": "raw_vs_product_bin_sensitivity",
            "item_count_bin": "changed bin assignment",
            "receipts_n": len(changed_bin_ids),
            "changed_bin_receipt_ids": ";".join(changed_bin_ids),
            "legacy_transcribed_count_receipts_n": sum(receipt_info[rid]["source_dataset"] == "legacy_reference_v3_1" for rid in changed_bin_ids),
            "Receipy_full_ledger_candidate_receipts_n": sum(receipt_info[rid]["source_dataset"] == "receipy_references_v2" for rid in changed_bin_ids),
            "unknown_inventory_receipts_excluded_n": sum(not info["item_count_usable_for_bins"] for info in receipt_info.values()),
        }
    )
    for row in count_distribution:
        row.setdefault("changed_bin_receipt_ids", "")
    write_csv(DEST / "item_count_distribution_summary.csv", count_distribution)

    # Compact reproducibility ledger.
    write_csv(DEST / "verified_input_hashes.csv", verified)
    countable_score_ids = {
        score_id for score_id, info in receipt_info.items() if info["item_count_usable_for_bins"]
    }
    countable_reference_ids = {
        str(receipt_info[score_id]["reference_id_join_key"]) for score_id in countable_score_ids
    }
    unknown_total_label_views = {
        (row["receipt_id"], row["condition"])
        for row in selected if row.get("reference_known") != "1"
    }
    summary = {
        "status": "completed",
        "saved_input_hashes_verified": len(verified),
        "selected_receipt_pairs": len(mapped_ids),
        "selected_total_score_rows": len(selected),
        "routes": ROUTES,
        "conditions": CONDITIONS,
        "reference_join": "legacy_ prefix stripped to receipt_*; new_* exact match; all known totals numerically agree with the joined frozen reference; one-to-one 50 receipt mapping",
        "receipt_count_proxy_n": len(countable_reference_ids),
        "item_count_unavailable_or_incomplete_receipts": sorted(set(mapped_ids) - countable_reference_ids),
        "unknown_total_label_views_n": len(unknown_total_label_views),
        "unknown_total_route_decisions_n": sum(row.get("reference_known") != "1" for row in selected),
        "item_count_bins": ["1-3", "4-5", "6+"],
        "binning_unit": "annotated product reference rows per receipt; raw-entry sensitivity also reported",
        "unknown_or_incomplete_item_counts_excluded_from_bins": True,
        "name_length_n": len(name_rows),
        "interpretation_limits": [
            "All receipt labels are assistant-audited/unverified.",
            "Legacy line-item inventory completeness is not independently established; its count is a transcribed-row proxy.",
            "Receipy product count uses row_type=product; raw entry count also includes promo/discount/adjustment entries.",
            "Product-name text length is not a full receipt transcription, OCR token count, or causal difficulty measure.",
            "Small item-count strata are descriptive and not inferential evidence of a causal item-count effect.",
        ],
    }
    (DEST / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Build a compact full-model summary without hiding the tiny middle bin.
    product_cells = {
        (row["route"], row["condition"], row["item_count_bin"]): row
        for row in stratified
        if row["count_basis"] == "product_item_count"
        and row["item_count_bin"] in {"1-3", "4-5", "6+"}
    }
    def cell_text(route: str, condition: str, bin_name: str) -> str:
        row = product_cells[(route, condition, bin_name)]
        n = int(row["known_total_label_n"])
        if not n:
            return "n/a"
        tp = int(row["exact_total_matches_n"])
        return f"{tp}/{n} ({100 * tp / n:.1f}%)"

    result_table = [
        "| Route | Clear 1–3 | Clear 4–5 | Clear 6+ | Scrunched 1–3 | Scrunched 4–5 | Scrunched 6+ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        result_table.append(
            "| " + route + " | " + " | ".join(
                [cell_text(route, "clear", b) for b in ("1-3", "4-5", "6+")]
                + [cell_text(route, "scrunched", b) for b in ("1-3", "4-5", "6+")]
            ) + " |"
        )

    product_bins = {row["item_count_bin"]: int(row["receipts_n"]) for row in count_distribution if row["count_basis"] == "product_item_count"}
    raw_bins = {row["item_count_bin"]: int(row["receipts_n"]) for row in count_distribution if row["count_basis"] == "raw_reference_entry_count"}
    name_combined = next(row for row in length_summary if row["source_dataset"] == "combined")
    volume_combined = next(row for row in receipt_length_summary if row["source_dataset"] == "combined")

    # A compact narrative with exact cells is useful to the report author.
    lines = [
        "# Receipt total accuracy by annotated product-row count",
        "",
        "This is a frozen-input, CPU-only descriptive join. The script verifies SHA-256 hashes recorded in the parent analysis manifest, maps `legacy_receipt_###` to `receipt_###` and `new_###` directly, and confirms that each known saved total matches the joined reference value. The 50 selected receipt IDs map one-to-one; 600 completed score rows cover six routes by two views.",
        "",
        f"Item-count proxy is available for {len(countable_reference_ids)}/50 selected receipts. One partial/fragment reference is kept in the per-receipt file and shown in a separate unbinned row. One of 100 total-view targets is unknown (six route decisions); unknown labels stay out of accuracy denominators and remain counted in coverage fields.",
        "",
        f"Product-count bin coverage is 1–3: {product_bins['1-3']}, 4–5: {product_bins['4-5']}, and 6+: {product_bins['6+']} receipts. Under raw reference-entry counts it is 1–3: {raw_bins['1-3']}, 4–5: {raw_bins['4-5']}, and 6+: {raw_bins['6+']}; two Receipy ledgers move bins (`new_005`, `new_013`) because raw entries include promotions/discounts. The 4–5 product bin has only two receipts, so its percentages are especially unstable. Legacy rows lack a `row_type`, so every listed `line_items` row is treated as a product for this proxy; full inventory completeness is not independently asserted.",
        "",
        "Exact total matches / known reference labels by route and condition (Wilson intervals and coverage counts are in the CSV):",
        "",
        *result_table,
        "",
        f"Across {int(name_combined['product_name_strings_n'])} readable/score-eligible product-name strings, length median is {int(name_combined['median_chars'])} characters (IQR {int(name_combined['p25_chars'])}–{int(name_combined['p75_chars'])}; range {int(name_combined['min_chars'])}–{int(name_combined['max_chars'])}). The sum of available product-name characters per receipt has median {int(volume_combined['median_known_product_name_chars_per_receipt'])} (IQR {int(volume_combined['p25_chars'])}–{int(volume_combined['p75_chars'])}; range {int(volume_combined['min_chars'])}–{int(volume_combined['max_chars'])}) across {volume_combined['receipts_n']} binned receipts. This is an item-list text-volume proxy, not full receipt text length or OCR token count.",
        "",
        "`per_receipt_total_item_counts.csv` retains item counts, coverage status, unknown item fields, unknown total-label status, and each route's saved match flag. `product_name_length_summary.csv` describes individual name lengths; `receipt_item_name_text_volume_proxy.csv` sums those characters per receipt. These are item-list text proxies only; no complete receipt transcription or OCR-token count exists in these references. `item_count_distribution_summary.csv` records raw-versus-product bin sensitivity. These descriptive strata do not establish that more items cause worse accuracy. The labels are assistant-audited and unverified.",
        "",
        "Reproduce from the analysis handoff directory with:",
        "",
        "```powershell",
        r"..\.venv-pilot\Scripts\python.exe .\stratify_totals_by_item_count.py",
        "```",
        "",
    ]
    (DEST / "README.md").write_text("\n".join(lines), encoding="utf-8")
    output_manifest = []
    for path in sorted(p for p in DEST.iterdir() if p.is_file() and p.name != "output_manifest.csv"):
        output_manifest.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(DEST / "output_manifest.csv", output_manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
