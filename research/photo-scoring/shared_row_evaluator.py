"""CPU-only scorer for the hash-frozen Receipy v2 image-only run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FREEZE = HERE / "frozen_v2"
PRED = HERE / "predictions_v2" / "primary"
OUT = HERE / "scoring_v2"
BOOTSTRAPS = 10_000
SEED = 271828

ROUTES = {
    "Tesseract+rules": ("ocr_total", "ocr_rows", "tesseract_rules"),
    "LayoutLMv3": ("layout_total", "layout_rows", "layoutlmv3"),
    "TrOCR+LayoutLMv3": ("cascade_total", "cascade_rows", "trocr_layoutlmv3"),
    "Donut": ("donut_total", "donut_rows", "donut"),
}
ITEM_TYPES = {"product", "promo product", "modifier", "modifier/add-on", "product/option group"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_freeze():
    gate_path = FREEZE / "reference_freeze_manifest.json"
    pair_path = FREEZE / "pairing_manifest.json"
    ref_path = FREEZE / "references.json"
    audit_path = FREEZE / "audit_provenance.json"
    gate = read_json(gate_path)
    if gate.get("audit_status") != "adjudicated":
        raise RuntimeError("No adjudicated audit gate; refusing to score")
    for key, path in (("pairing_manifest_sha256", pair_path), ("reference_sha256", ref_path), ("independent_audit_sha256", audit_path)):
        if gate.get(key) != sha256(path):
            raise RuntimeError(f"Frozen v2 hash mismatch: {key}")
    pairing = read_json(pair_path)
    refs = read_json(ref_path)
    if len(pairing.get("primary_pairs", [])) != 16 or len(pairing.get("records", [])) != 32:
        raise RuntimeError("V2 primary cohort does not have 16 pairs / 32 images")
    return gate, pairing, refs


def strict_aud_amount(value):
    """Parse one AUD-like scalar; labels may surround it, but no second number is guessed."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    s = str(value).strip()
    if not s:
        return None
    # A malformed token such as 4:000, 25-974.00, or a multi-value field yields
    # multiple scalar tokens or fails comma grouping and is rejected.
    token_re = re.compile(r"(?<![A-Za-z0-9])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)(?![A-Za-z0-9])")
    matches = list(token_re.finditer(s))
    if len(matches) != 1:
        return None
    token = matches[0].group(1)
    if re.search(r"\d\s+\d{3}(?!\d)", s):
        return None
    if token.count(".") > 1 or token.startswith(".") or token.endswith("."):
        return None
    try:
        return Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None


def normalized_name(value):
    if value is None:
        return None
    text = unicodedata.normalize("NFKD", str(value)).casefold()
    norm = "".join(ch for ch in text if ch.isalnum())
    return norm or None


def normalized_quantity(value):
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"^[xX]\s*", "", s)
    s = re.sub(r"\s*[xX]$", "", s)
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)(?:\s*([A-Za-z%][A-Za-z0-9%/-]*))?", s)
    if not m:
        return None
    try:
        number = Decimal(m.group(1))
    except InvalidOperation:
        return None
    unit = (m.group(2) or "").casefold()
    return (number, unit)


def norm_value(field, value):
    if field == "nm":
        return normalized_name(value)
    if field == "cnt":
        return normalized_quantity(value)
    if field in ("price", "unitprice"):
        return strict_aud_amount(value)
    return None


def clean_predictions(rows):
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and any(r.get(k) not in (None, "") for k in ("nm", "cnt", "price", "unitprice", "itemsubtotal"))]


def target_rows(reference):
    targets = []
    for item in reference.get("items", []):
        if item.get("row_type") not in ITEM_TYPES:
            continue
        target = {}
        name = item.get("name")
        notes = (item.get("notes") or "").casefold()
        name_unclear = any(tok in (name or "") for tok in ("?", "�")) or any(tok in notes for tok in ("partly unclear", "unreadable", "text clipped"))
        if item.get("name_score_eligible") and not name_unclear:
            target["nm"] = name
        if item.get("quantity") not in (None, ""):
            target["cnt"] = item["quantity"]
        if item.get("line_total") not in (None, ""):
            target["price"] = item["line_total"]
        if item.get("unit_price") not in (None, ""):
            target["unitprice"] = item["unit_price"]
        if target:
            targets.append(target)
    return targets


def score_rows(gt_rows, pred_rows, support=("nm", "cnt", "price", "unitprice")):
    targets = [{k: v for k, v in row.items() if k in support and v not in (None, "")} for row in gt_rows]
    targets = [r for r in targets if r]
    preds = clean_predictions(pred_rows)
    adjacency = []
    for pred in preds:
        edges = []
        for gi, truth in enumerate(targets):
            if all(pred.get(field) not in (None, "") and norm_value(field, pred[field]) is not None and norm_value(field, pred[field]) == norm_value(field, value) for field, value in truth.items()):
                edges.append(gi)
        adjacency.append(edges)
    owner = {}

    def augment(pi, seen):
        for gi in adjacency[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in owner or augment(owner[gi], seen):
                owner[gi] = pi
                return True
        return False

    tp = sum(augment(pi, set()) for pi in range(len(preds)))
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(targets) - tp, "gt_rows": len(targets), "pred_rows": len(preds)}


def field_score(gt_rows, pred_rows, field):
    targets = [r[field] for r in gt_rows if r.get(field) not in (None, "")]
    preds = [r.get(field) for r in clean_predictions(pred_rows) if r.get(field) not in (None, "")]
    adjacency = [[i for i, truth in enumerate(targets) if norm_value(field, pred) is not None and norm_value(field, pred) == norm_value(field, truth)] for pred in preds]
    owner = {}

    def augment(pi, seen):
        for gi in adjacency[pi]:
            if gi in seen:
                continue
            seen.add(gi)
            if gi not in owner or augment(owner[gi], seen):
                owner[gi] = pi
                return True
        return False

    tp = sum(augment(pi, set()) for pi in range(len(preds)))
    return {"tp": tp, "fp": len(preds) - tp, "fn": len(targets) - tp, "gt_fields": len(targets), "pred_fields": len(preds)}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def wilson(k, n, z=1.959963984540054):
    if not n:
        return None, None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def paired_bootstrap_delta(by_id, ids, metric, draws, seed_offset=0):
    rng = random.Random(SEED + seed_offset)
    vals = []
    for _ in range(draws):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        a = metric([by_id[x]["clear"] for x in sample])
        b = metric([by_id[x]["scrunched"] for x in sample])
        vals.append(b - a)
    vals.sort()
    return vals[int(0.025 * draws)], vals[min(draws - 1, int(0.975 * draws))]


def binary_metric(xs):
    return sum(xs) / len(xs) if xs else 0.0


def f1_metric(xs):
    tp = sum(x["tp"] for x in xs)
    fp = sum(x["fp"] for x in xs)
    fn = sum(x["fn"] for x in xs)
    return prf(tp, fp, fn)[2]


def exact_mcnemar(clear_only, scrunched_only):
    n = clear_only + scrunched_only
    if n == 0:
        return 1.0
    low = min(clear_only, scrunched_only)
    tail = sum(math.comb(n, k) for k in range(low + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm_adjust(pvalues):
    m = len(pvalues)
    ordered = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * m
    running = 0.0
    for rank, (idx, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[idx] = running
    return adjusted


def main():
    gate, pairing, references = verify_freeze()
    pair_ids = [x["receipt_id"] for x in pairing["primary_pairs"]]
    refs = {x["receipt_id"]: x for x in references["records"] if x["receipt_id"] in set(pair_ids)}
    if set(refs) != set(pair_ids):
        raise RuntimeError("Primary reference records do not match the 16 confirmed receipt IDs")
    image_records = {(r["receipt_id"], r["condition"]): r for r in pairing["records"]}
    output_cache = {}
    for rid in pair_ids:
        for cond in ("clear", "scrunched"):
            rec = image_records[(rid, cond)]
            path = PRED / f"{rid}__{cond}.json"
            if path.exists():
                obj = read_json(path)
                if obj.get("source_sha256") != rec["sha256"]:
                    raise RuntimeError(f"Prediction source hash mismatch: {path.name}")
                output_cache[(rid, cond)] = obj
            else:
                output_cache[(rid, cond)] = {"receipt_id": rid, "condition": cond, "status": "missing_result", "model_status": {}}

    totals = []
    row_receipts = []
    field_receipts = []
    raw_per_receipt = defaultdict(lambda: defaultdict(dict))
    name_field_count = quantity_field_count = line_amount_count = unit_amount_count = 0
    for rid in pair_ids:
        ref = refs[rid]
        complete_inventory = bool(ref.get("complete_visible_inventory_candidate"))
        if not complete_inventory:
            continue
        targets = target_rows(ref)
        for row in targets:
            name_field_count += int("nm" in row)
            quantity_field_count += int("cnt" in row)
            line_amount_count += int("price" in row)
            unit_amount_count += int("unitprice" in row)
        for cond in ("clear", "scrunched"):
            out = output_cache[(rid, cond)]
            for route, (total_key, rows_key, status_key) in ROUTES.items():
                raw = out.get(total_key)
                parsed = strict_aud_amount(raw)
                truth = strict_aud_amount(ref["total"])
                status = (out.get("model_status") or {}).get(status_key, {}).get("status", "no_route_status")
                exact = int(parsed is not None and truth is not None and parsed == truth)
                totals.append({
                    "receipt_id": rid, "condition": cond, "route": route,
                    "reference_total": ref["total"], "raw_prediction": raw,
                    "parsed_prediction": str(parsed) if parsed is not None else None,
                    "parseable": int(parsed is not None), "exact": exact,
                    "run_status": out.get("status", "missing_result"), "route_status": status,
                    "prediction_file": str(PRED / f"{rid}__{cond}.json"),
                })
                raw_rows = out.get(rows_key) or []
                if not complete_inventory:
                    continue
                row_summary = score_rows(targets, raw_rows)
                row_summary.update({
                    "receipt_id": rid, "condition": cond, "route": route,
                    "run_status": out.get("status", "missing_result"), "route_status": status,
                })
                row_receipts.append(row_summary)
                for field in ("nm", "cnt", "price", "unitprice"):
                    fs = field_score(targets, raw_rows, field)
                    fs.update({"receipt_id": rid, "condition": cond, "route": route, "field": field})
                    field_receipts.append(fs)
                raw_per_receipt[route][rid][cond] = {"exact_total": exact, **row_summary}

    # The reference set marks new_014 as an incomplete photo fragment. It remains
    # in total decisions, but is not part of full row PRF1.
    for rid in pair_ids:
        if refs[rid].get("complete_visible_inventory_candidate"):
            continue
        for cond in ("clear", "scrunched"):
            out = output_cache[(rid, cond)]
            for route, (total_key, _rows_key, status_key) in ROUTES.items():
                raw = out.get(total_key)
                parsed = strict_aud_amount(raw)
                truth = strict_aud_amount(refs[rid]["total"])
                status = (out.get("model_status") or {}).get(status_key, {}).get("status", "no_route_status")
                totals.append({"receipt_id": rid, "condition": cond, "route": route,
                    "reference_total": refs[rid]["total"], "raw_prediction": raw,
                    "parsed_prediction": str(parsed) if parsed is not None else None,
                    "parseable": int(parsed is not None), "exact": int(parsed is not None and truth is not None and parsed == truth),
                    "run_status": out.get("status", "missing_result"), "route_status": status,
                    "prediction_file": str(PRED / f"{rid}__{cond}.json"), "row_scope": "excluded: partial photograph fragment"})

    total_summary = []
    paired_changes = []
    pvals = []
    for ix, route in enumerate(ROUTES):
        route_total = [x for x in totals if x["route"] == route]
        by_id = {rid: {cond: next(x for x in route_total if x["receipt_id"] == rid and x["condition"] == cond) for cond in ("clear", "scrunched")} for rid in pair_ids}
        for cond in ("clear", "scrunched"):
            vals = [by_id[rid][cond]["exact"] for rid in pair_ids]
            k, n = sum(vals), len(vals)
            lo, hi = wilson(k, n)
            total_summary.append({"route": route, "condition": cond, "correct": k, "n_receipts": n,
                "accuracy": k / n if n else None, "wilson95_low": lo, "wilson95_high": hi,
                "parseable": sum(by_id[rid][cond]["parseable"] for rid in pair_ids),
                "errors_or_timeouts": sum(by_id[rid][cond]["run_status"] not in ("complete", "partial") or by_id[rid][cond]["route_status"] != "ok" for rid in pair_ids)})
        transitions = {"both_correct": 0, "clear_only": 0, "scrunched_only": 0, "neither": 0}
        delta_by = {}
        for rid in pair_ids:
            c = by_id[rid]["clear"]["exact"]
            s = by_id[rid]["scrunched"]["exact"]
            if c and s: transitions["both_correct"] += 1
            elif c: transitions["clear_only"] += 1
            elif s: transitions["scrunched_only"] += 1
            else: transitions["neither"] += 1
            delta_by[rid] = {"clear": c, "scrunched": s}
        p = exact_mcnemar(transitions["clear_only"], transitions["scrunched_only"])
        pvals.append(p)
        lo, hi = paired_bootstrap_delta(delta_by, pair_ids, binary_metric, BOOTSTRAPS, ix)
        paired_changes.append({"route": route, **transitions,
            "scrunched_minus_clear_pp": 100 * (transitions["scrunched_only"] - transitions["clear_only"]) / len(pair_ids),
            "paired_bootstrap95_low_pp": 100 * lo, "paired_bootstrap95_high_pp": 100 * hi,
            "exact_mcnemar_p": p})
    adjusted = holm_adjust(pvals)
    for row, q in zip(paired_changes, adjusted):
        row["holm_adjusted_p"] = q

    row_summaries = []
    by_route_condition = {}
    for route in ROUTES:
        for cond in ("clear", "scrunched"):
            xs = [x for x in row_receipts if x["route"] == route and x["condition"] == cond]
            tp, fp, fn = (sum(x[k] for x in xs) for k in ("tp", "fp", "fn"))
            precision, recall, f1 = prf(tp, fp, fn)
            all_correct = [int(x["tp"] == x["gt_rows"] and x["fp"] == 0 and x["fn"] == 0) for x in xs]
            ac_lo, ac_hi = wilson(sum(all_correct), len(all_correct))
            by_id = {x["receipt_id"]: x for x in xs}
            row_summaries.append({"route": route, "condition": cond,
                "complete_visible_inventories": len(xs), "eligible_gt_rows": sum(x["gt_rows"] for x in xs),
                "predicted_rows": sum(x["pred_rows"] for x in xs), "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "all_rows_correct_receipts": sum(all_correct), "all_rows_correct_n": len(xs),
                "all_rows_correct_rate": sum(all_correct) / len(xs) if xs else None,
                "all_rows_correct_wilson95_low": ac_lo, "all_rows_correct_wilson95_high": ac_hi,
                "cluster_bootstrap_f1_95_low": None, "cluster_bootstrap_f1_95_high": None})
            by_route_condition[(route, cond)] = by_id

    # Bootstrap row F1 intervals and paired clean-to-scrunched F1 changes over
    # receipt identities (not over rows or photographs).
    row_paired = []
    row_cursor = 0
    for route_ix, route in enumerate(ROUTES):
        ids = [rid for rid in pair_ids if refs[rid].get("complete_visible_inventory_candidate")]
        if not ids:
            continue
        def receipt_f1(xs):
            return f1_metric(xs)
        grouped = {cond: [by_route_condition[(route, cond)][rid] for rid in ids] for cond in ("clear", "scrunched")}
        rng = random.Random(SEED + 100 + route_ix)
        samples = {"clear": [], "scrunched": [], "delta": []}
        for _ in range(BOOTSTRAPS):
            pick = [ids[rng.randrange(len(ids))] for _ in ids]
            c_stats = [by_route_condition[(route, "clear")][rid] for rid in pick]
            s_stats = [by_route_condition[(route, "scrunched")][rid] for rid in pick]
            c_f1 = receipt_f1(c_stats)
            s_f1 = receipt_f1(s_stats)
            samples["clear"].append(c_f1)
            samples["scrunched"].append(s_f1)
            samples["delta"].append(s_f1 - c_f1)
        for cond in ("clear", "scrunched"):
            vals = sorted(samples[cond])
            row = next(x for x in row_summaries if x["route"] == route and x["condition"] == cond)
            row["cluster_bootstrap_f1_95_low"] = vals[int(0.025 * BOOTSTRAPS)]
            row["cluster_bootstrap_f1_95_high"] = vals[int(0.975 * BOOTSTRAPS)]
        d = sorted(samples["delta"])
        row_paired.append({"route": route,
            "scrunched_minus_clear_f1": row_next(row_summaries, route, "scrunched", "f1") - row_next(row_summaries, route, "clear", "f1"),
            "paired_cluster_bootstrap95_low": d[int(0.025 * BOOTSTRAPS)],
            "paired_cluster_bootstrap95_high": d[int(0.975 * BOOTSTRAPS)],
            "receipt_clusters": len(ids), "bootstrap_draws": BOOTSTRAPS, "seed": SEED + 100 + route_ix})

    field_summaries = []
    for route in ROUTES:
        for cond in ("clear", "scrunched"):
            for field in ("nm", "cnt", "price", "unitprice"):
                xs = [x for x in field_receipts if x["route"] == route and x["condition"] == cond and x["field"] == field]
                tp, fp, fn = (sum(x[k] for x in xs) for k in ("tp", "fp", "fn"))
                precision, recall, f1 = prf(tp, fp, fn)
                field_summaries.append({"route": route, "condition": cond, "field": field,
                    "eligible_gt_fields": sum(x["gt_fields"] for x in xs), "predicted_fields": sum(x["pred_fields"] for x in xs),
                    "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})

    write_csv(OUT / "total_decisions.csv", totals)
    write_csv(OUT / "total_summary.csv", total_summary)
    write_csv(OUT / "paired_total_changes.csv", paired_changes)
    write_csv(OUT / "row_receipt_scores.csv", row_receipts)
    write_csv(OUT / "row_summary.csv", row_summaries)
    write_csv(OUT / "paired_row_f1_changes.csv", row_paired)
    write_csv(OUT / "field_retrieval.csv", field_summaries)
    write_csv(OUT / "field_eligibility.csv", [{
        "eligible_name_targets": name_field_count, "eligible_quantity_targets": quantity_field_count,
        "eligible_printed_line_total_targets": line_amount_count, "eligible_printed_unit_price_targets": unit_amount_count,
        "scope": "15 complete-visible receipt ledgers; excludes new_014 fragment and both new_010 unreadable unpaired images",
    }])
    audit = {
        "scorer_version": "receipy_shared_policy_v2",
        "freeze_gate_sha256": sha256(FREEZE / "reference_freeze_manifest.json"),
        "pairing_manifest_sha256": sha256(FREEZE / "pairing_manifest.json"),
        "reference_sha256": sha256(FREEZE / "references.json"),
        "primary_receipts": pair_ids,
        "expected_model_views": 16 * 2 * len(ROUTES),
        "completed_prediction_files": sum((PRED / f"{rid}__{c}.json").exists() for rid in pair_ids for c in ("clear", "scrunched")),
        "model_routes": list(ROUTES),
        "row_inventory_scope": "15 full photographed visible ledgers; new_014 fragment excluded from row PRF1; 58 item/option row candidates before field unknown/confidence filtering",
        "strict_unknown_and_adjustment_policy": "unknown fields excluded; single_amount excluded from typed row amounts; promotion/discount/tender/tax records excluded from item-row scores; product/option and modifier rows retained",
        "bootstrap_draws": BOOTSTRAPS,
        "bootstrap_seed": SEED,
        "human_verification": "pending; results remain assistant-audited exploratory evidence",
    }
    write_json(OUT / "score_manifest.json", audit)
    print(json.dumps({"output": str(OUT), "total_decisions": len(totals), "primary_receipts": len(pair_ids),
        "prediction_files_present": audit["completed_prediction_files"], "row_receipts": len(row_receipts),
        "eligible_fields": {"name": name_field_count, "quantity": quantity_field_count, "line_total": line_amount_count, "unit_price": unit_amount_count}}, indent=2))


def row_next(rows, route, cond, key):
    return next(x[key] for x in rows if x["route"] == route and x["condition"] == cond)


if __name__ == "__main__":
    main()
