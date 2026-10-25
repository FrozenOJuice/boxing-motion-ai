"""Read existing preparation, review and result metadata without running pipelines."""

from collections import Counter
from functools import wraps

from boxing_motion_ai.data.annotation_review import LABELS, RECORDINGS, ReviewLedger, digest
from boxing_motion_ai.data.curation import CurationConfig, latest

from .artifacts import MetadataStore, ServiceError, strict_json
from .contracts import Catalogue, QueryResult


def query(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            value = function(*args, **kwargs)
            return (
                value
                if isinstance(value, QueryResult)
                else QueryResult(status="available", data=value)
            )
        except ServiceError as exc:
            return QueryResult(status=exc.status, issues=[exc.code])
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, RecursionError):
            return QueryResult(status="invalid", issues=["malformed_or_unsupported_metadata"])

    return wrapped


def require(condition, code, status="invalid"):
    if not condition:
        raise ServiceError(status, code)


class DashboardService:
    """An explicitly registered development subset, not a raw filesystem inventory.

    Methods accept IDs, never caller-supplied artifact paths. Construction takes a
    trusted application root and catalogue path. The catalogue is a process snapshot;
    registered hashes are rechecked on every query. Restart to adopt a new catalogue.
    Pose arrays, media, checkpoints and the preservation manifest are never opened.
    """

    def __init__(self, root, catalogue="configs/dashboard_catalogue.json"):
        self.store = MetadataStore(root)
        value, self.catalogue_sha256 = self.store.json(catalogue)
        self.catalogue = Catalogue.model_validate(value)

    def _config(self):
        ref = self.catalogue.curation
        value, _ = self.store.json(ref.path, ref.sha256)
        return CurationConfig.model_validate(value)

    def _recording(self, recording_id):
        require(recording_id in RECORDINGS, "unregistered_recording")
        return self._config().sources[recording_id]

    def _preparation(self, recording_id, source):
        value, _ = self.store.json(
            f"{source.dataset}/preparation.json", source.inputs["preparation.json"]
        )
        require(value.get("video_id") == recording_id, "recording_identity_mismatch")
        expected = "v5-preparation-1.0" if recording_id == "V5" else "recording-preparation-1.1"
        require(value.get("schema_version") == expected, "unsupported_preparation_schema")
        require(value.get("npz_sha256") == source.inputs["poses.npz"], "pose_reference_mismatch")
        return value

    def _rows(self, recording_id, source):
        raw, _ = self.store.read(
            f"{source.dataset}/candidates.jsonl", source.inputs["candidates.jsonl"]
        )
        rows = [strict_json(line) for line in raw.splitlines() if line.strip()]
        seen = set()
        expected = "v5-candidate-1.0" if recording_id == "V5" else "recording-candidate-1.1"
        for row in rows:
            key = row.get("source_row")
            require(type(key) is int and key > 0 and key not in seen, "invalid_source_row")
            require(row.get("video_id") == recording_id, "recording_identity_mismatch")
            require(row.get("schema_version") == expected, "unsupported_candidate_schema")
            # Reversed/unresolved intervals must remain visible; never repair or hide them.
            require(
                all(k in row for k in ("raw_start", "raw_end", "punch_type_provisional")),
                "candidate_fields_missing",
                "incomplete",
            )
            seen.add(key)
        return rows

    def _policy(self, recording_id, source, rows):
        value, _ = self.store.json(source.policy.path, source.policy.sha256)
        require(
            value.get("schema_version") == "recording-mask-policy-index-1.1"
            and value.get("video_id") == recording_id
            and value.get("policy") == "mask_aware70"
            and value.get("source_mask_threshold") == 0.5
            and value.get("inputs") == source.inputs,
            "policy_binding_mismatch",
        )
        selected = value["selected_source_rows"]
        require(
            all(type(r) is int for r in selected)
            and len(selected) == len(set(selected))
            and set(selected) <= {r["source_row"] for r in rows},
            "invalid_policy_selection",
        )
        return value

    def _ledger(self, config):
        ref = config.review_ledger
        value, _ = self.store.json(ref.path, ref.sha256)
        ledger = ReviewLedger.model_validate(value)
        require(
            ledger.cases_sha256 == config.cases.sha256
            and ledger.diagnostic_sha256 == config.diagnostic.sha256,
            "review_evidence_binding_mismatch",
            "stale",
        )
        return ledger

    @query
    def recording(self, recording_id: str):
        source = self._recording(recording_id)
        prep = self._preparation(recording_id, source)
        require(
            type(prep.get("source_frame_count")) is int and prep["source_frame_count"] >= 0,
            "frame_inventory_missing_or_invalid",
            "incomplete",
        )
        return {
            "recording_id": recording_id,
            "role": source.role,
            "source_group": source.source_group,
            "related_recordings_guard": source.related_recordings_guard,
            "boxer_id": prep.get("boxer_id"),
            "session_id": prep.get("session_id"),
            "preparation": prep,
            "pose_array": {
                "path": f"{source.dataset}/poses.npz",
                "expected_sha256": source.inputs["poses.npz"],
                "verified": False,
            },
            "annotation_status": "provisional",
        }

    @query
    def inventory(self):
        records = {v: self.recording(v).model_dump(mode="json") for v in RECORDINGS}
        data = {
            "dataset_id": "boxingvi",
            "scope": "six_registered_approved_preparations_not_raw_dataset_discovery",
            "records": records,
            "available_recordings": sum(r["status"] == "available" for r in records.values()),
            "unregistered_recordings": ["V1", "V2", "V3"],
            "reserved_recordings": ["V9"],
            "catalogue_sha256": self.catalogue_sha256,
        }
        complete = all(r["status"] == "available" for r in records.values())
        return QueryResult(
            status="available" if complete else "incomplete",
            data=data,
            issues=[] if complete else ["recording_metadata_unavailable"],
        )

    @query
    def annotations(self, recording_id: str, *, offset: int = 0, limit: int = 50):
        require(type(offset) is int and offset >= 0, "invalid_offset")
        require(type(limit) is int and 1 <= limit <= 100, "invalid_limit")
        source = self._recording(recording_id)
        rows = self._rows(recording_id, source)
        return {
            "recording_id": recording_id,
            "total": len(rows),
            "offset": offset,
            "items": rows[offset : offset + limit],
        }

    @query
    def annotation(self, recording_id: str, source_row: int):
        require(type(source_row) is int and source_row > 0, "invalid_source_row")
        source = self._recording(recording_id)
        rows = self._rows(recording_id, source)
        row = next((r for r in rows if r["source_row"] == source_row), None)
        require(row is not None, "annotation_not_found", "missing")
        return {"source": row, "source_candidate_sha256": digest(row)}

    @query
    def statistics(self, recording_id: str):
        source = self._recording(recording_id)
        rows = self._rows(recording_id, source)
        prep = self._preparation(recording_id, source)
        policy = self._policy(recording_id, source, rows)
        selected = set(policy["selected_source_rows"])
        counts = Counter(r["punch_type_provisional"] for r in rows if r["source_row"] in selected)
        require(set(counts) <= set(LABELS), "unknown_selected_class")
        return {
            "recording_id": recording_id,
            "role": source.role,
            "source_group": source.source_group,
            "related_recordings_guard": source.related_recordings_guard,
            "annotation_rows": len(rows),
            "approved_provisional_examples": len(selected),
            "approved_by_class": {label: counts[label] for label in LABELS},
            "absent_approved_classes": [label for label in LABELS if not counts[label]],
            "unselected_rows": [r["source_row"] for r in rows if r["source_row"] not in selected],
            "preparation_class_counts": prep.get("class_counts"),
            "policy": policy,
            "background": "unlabeled_is_unknown_not_no_punch",
            "source_pose_bytes_verified": False,
        }

    @query
    def review(self, recording_id: str, source_row: int):
        source = self._recording(recording_id)
        require(type(source_row) is int and source_row > 0, "invalid_source_row")
        config = self._config()
        ledger = self._ledger(config)
        key = f"{recording_id}:{source_row}"
        case = ledger.cases.get(key)
        require(case is not None, "review_not_found", "missing")
        row = next(
            (r for r in self._rows(recording_id, source) if r["source_row"] == source_row), None
        )
        require(
            row is not None and digest(row) == case.source_candidate_sha256,
            "review_candidate_changed",
            "stale",
        )
        revision = latest(ledger, key)
        return {
            "id": key,
            "source": row,
            "ledger_version": ledger.ledger_version,
            "case": case.model_dump(mode="json"),
            "latest": revision.model_dump(mode="json") if revision else None,
            "human_accepted": bool(
                revision and revision.actor_type == "human" and revision.disposition == "accept"
            ),
        }

    @query
    def review_queue(self):
        config = self._config()
        ledger = self._ledger(config)
        entries = []
        for priority, ids in (
            ("dispute", config.priority_disputes),
            ("anchor", config.verification_anchors),
        ):
            for case_id in ids:
                require(case_id in ledger.cases, "queue_case_missing", "incomplete")
                recording, row = case_id.split(":")
                entries.append(
                    {
                        "id": case_id,
                        "priority": priority,
                        "review": self.review(recording, int(row)).model_dump(mode="json"),
                    }
                )
        complete = all(entry["review"]["status"] == "available" for entry in entries)
        return QueryResult(
            status="available" if complete else "incomplete",
            issues=[] if complete else ["review_metadata_unavailable"],
            data={
                "ledger_version": ledger.ledger_version,
                "entries": entries,
                "confirmation_policy": "assistant_observations_are_not_human_certification",
            },
        )

    @query
    def review_status(self):
        ledger = self._ledger(self._config())
        revisions = [latest(ledger, key) for key in ledger.cases]
        dispositions = Counter(r.disposition for r in revisions if r)
        return {
            "ledger_version": ledger.ledger_version,
            "case_ids": list(ledger.cases),
            "case_count": len(ledger.cases),
            "dispositions": dict(dispositions),
            "human_accepted": sum(
                bool(r and r.actor_type == "human" and r.disposition == "accept") for r in revisions
            ),
            "without_revision": sum(r is None for r in revisions),
            "source_candidate_fingerprints_verified": False,
        }

    @query
    def review_evidence(self, recording_id: str, source_row: int):
        self._recording(recording_id)  # Reject V9 before opening evidence.
        require(type(source_row) is int and source_row > 0, "invalid_source_row")
        config = self._config()
        cases, _ = self.store.json(config.cases.path, config.cases.sha256)
        key = f"{recording_id}:{source_row}"
        require(isinstance(cases, list), "unsupported_evidence_schema")
        matches = [case for case in cases if case["id"] == key]
        require(len(matches) <= 1, "duplicate_evidence_case")
        require(bool(matches), "evidence_not_found", "missing")
        # Preserve native indices, rational PTS, masks, observations and predictions verbatim.
        return {"case": matches[0], "diagnostic_is_ground_truth": False}

    @query
    def curation_status(self):
        config = self._config()
        ledger = self._ledger(config)
        ref = self.catalogue.proposal
        value, _ = self.store.json(ref.path, ref.sha256)
        require(
            value.get("schema_version") == "curated-selection-proposal-1.0",
            "unsupported_curation_schema",
        )
        provenance = value.get("provenance", {})
        require(
            provenance.get("config_sha256") == self.catalogue.curation.sha256
            and provenance.get("ledger_sha256") == config.review_ledger.sha256
            and value.get("ledger_version") == ledger.ledger_version,
            "stale_curation_snapshot",
            "stale",
        )
        require(value.get("training_enabled") is False, "unsupported_curation_state")
        return {
            "proposal": value,
            "verification_scope": "snapshot_and_config_hashes_only",
            "preservation_manifest_reverified": False,
        }

    @query
    def experiments(self):
        return {
            "items": {
                key: {
                    "status": value.status,
                    "description": value.description,
                    "format": value.format,
                    "metric_ids": list(value.metrics),
                }
                for key, value in self.catalogue.experiments.items()
            }
        }

    @query
    def experiment(self, experiment_id: str):
        spec = self.catalogue.experiments.get(experiment_id)
        require(spec is not None, "unregistered_experiment")
        value, actual = self.store.json(spec.summary.path, spec.summary.sha256)
        require(isinstance(value, dict), "invalid_experiment_summary")
        if spec.format == "rgb-legacy-1":
            require(
                "schema_version" not in value and {"seed", "cases", "smoke"} <= value.keys(),
                "unsupported_experiment_schema",
            )
        else:
            require(value.get("schema_version") == spec.format, "unsupported_experiment_schema")
            required = {"seeds", "by_model"} if spec.format.startswith("velocity") else {"cases"}
            require(required <= value.keys(), "incomplete_experiment_summary", "incomplete")
        return {
            "id": experiment_id,
            "format": spec.format,
            "status": spec.status,
            "sha256": actual,
            "results": value,
            "interpretation": "stored_metrics_not_recomputed_or_boxer_independent",
        }

    @query
    def model_metrics(self, experiment_id: str, metric_id: str):
        spec = self.catalogue.experiments.get(experiment_id)
        require(spec is not None, "unregistered_experiment")
        ref = spec.metrics.get(metric_id)
        require(ref is not None, "unregistered_metrics")
        value, actual = self.store.json(ref.path, ref.sha256)
        absent = {}
        for partition in ("train", "development"):
            metrics = value[partition]
            require(metrics["labels"] == list(LABELS), "unsupported_metric_labels")
            for name in ("accuracy", "macro_f1"):
                require(
                    type(metrics[name]) in (int, float) and 0 <= metrics[name] <= 1,
                    "invalid_metric_value",
                )
            matrix = metrics["confusion_matrix"]
            require(
                len(matrix) == len(LABELS) and all(len(row) == len(LABELS) for row in matrix),
                "invalid_confusion_matrix",
            )
            for label, row in zip(LABELS, matrix, strict=True):
                scores = metrics["per_class"][label]
                for name in ("precision", "recall", "f1"):
                    require(
                        type(scores[name]) in (int, float) and 0 <= scores[name] <= 1,
                        "invalid_metric_value",
                    )
                require(
                    type(scores["support"]) is int and scores["support"] >= 0, "invalid_support"
                )
                require(all(type(n) is int and n >= 0 for n in row), "invalid_confusion_counts")
                require(sum(row) == metrics["per_class"][label]["support"], "support_mismatch")
            absent[partition] = [
                label for label in LABELS if metrics["per_class"][label]["support"] == 0
            ]
        return {
            "id": metric_id,
            "sha256": actual,
            "metrics": value,
            "absent_classes": absent,
            "metrics_recomputed": False,
        }
