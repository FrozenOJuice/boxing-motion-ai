# Phase 0 annotation contract and mapping

Contract version **1.0** lives in `src/boxing_motion_ai/models.py`. Dataset, video, event and mapping
records carry `schema_version`. Unknown boxer/session/annotator/source/timebase values stay null.
The report stores the supplied mapping and its SHA-256; the mapping's `evidence` is an operator
note, not independent proof of the authors' schema. Do not silently reinterpret a release.

## Events and background

Punch events use `jab`, `cross`, `lead_hook`, `rear_hook`, `lead_uppercut`, `rear_uppercut`, or
`unknown`. Preserve uncertain punch classes as `unknown` only with an explicit mapping.
`no_punch` is a separate background state, excluded from the event class enum. Explicitly review
negative footage before labeling it; gaps may contain punches, feints, defenses or missing labels.
Do not map a source's negative or excluded-action code into a punch class.

Events retain raw frame bounds, source class, workbook/sheet/row, annotation version and review
status. `frame_base` and `end_convention` are mandatory: `[start, end]` inclusive and `[start, end)`
exclusive differ by one frame. Validation converts temporarily to zero-based half-open intervals;
source bounds are never overwritten. Onset/peak are optional, initially null and unreviewed.
Source start/end does not establish biomechanical onset/peak. Future human labels must lie within
the event and peak cannot precede onset. Hand remains null unless independently annotated or
stance is established; left/right alone does not determine lead/rear. Do not invent identities.

Repeated video/bounds/class/boxer events are errors. Overlapping intervals are suspect and need
human review: they may be duplicates, different boxers or real annotation ambiguities. Coverage
uses the union of parsed intervals within an exact known media frame count, so overlaps cannot
inflate it. Unknown or out-of-bounds intervals are excluded. Coverage does not prove exhaustive
annotation or that the remaining frames are negative. Candidate groups are **not frozen splits**;
review repeated athletes and recordings across all files before assigning any splits.

## Explicit XLSX layout mapping

**The following is entirely synthetic, not the BoxingVI release schema.** Never copy its assumed
positions, class tokens or frame conventions without checking real workbooks and source media.
Column numbers and row numbers are one-based Excel positions. Each configured sheet currently
maps to exactly one explicit video; multiple workbooks/sheets can map to the same recording.
If real files embed video IDs per row or a different layout, stop and extend the adapter against
that evidence instead of forcing this format. No inferred filename matching is performed.

```json
{
  "schema_version": "1.0",
  "annotation_version": "synthetic-example-v1",
  "evidence": "Synthetic example only. Replace with inspected release and frame convention evidence.",
  "sheets": [
    {
      "workbook": "Annotation_files/example.xlsx",
      "sheet": "Punches",
      "data_start_row": 2,
      "columns": {"frame_start": 1, "frame_end": 2, "punch_type": 3},
      "video": "RGB_videos/example.mp4",
      "frame_base": 0,
      "end_convention": "inclusive",
      "class_map": {"J": "jab", "C": "cross", "uncertain": "unknown"},
      "boxer_id": null,
      "session_id": null
    }
  ]
}
```

Workbook/video paths must be dataset-root-relative POSIX paths; no absolute paths, traversal or
symlink files. Supply all event sheets; unmapped sheets and absent mapped sheets are diagnostics.
Class keys must exactly match stringified Excel cell values, including capitalization/whitespace.
Numeric class codes require author-supported mappings. Formula frame cells, text frames,
nonintegral/negative/reversed bounds, missing cells and unknown types are rejected, with source
locations retained. Formula values are not evaluated. All nonempty rows from `data_start_row`
onward are treated as candidate event rows; metadata/footer/merged layouts require inspection.
Unused columns are not assumed to contain IDs or timing labels.

Review at least the first/last labeled events, every class, multiple boxers/sessions/viewpoints,
ambiguous onsets, feints and negative motion. Confirm original recording versions, frame indexing,
end inclusivity and any resampling or trimming. Inspect variable frame rate and use timestamps
where needed; average FPS alone is insufficient. Record manual decisions and author evidence
before approving the Phase 0 gate. No review is claimed by the automated report.
