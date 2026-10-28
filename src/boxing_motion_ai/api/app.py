"""Thin localhost API over existing services; exact-frame media is opt-in per request."""

from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from boxing_motion_ai.data.annotation_review import LABELS, RECORDINGS
from boxing_motion_ai.services import DashboardService
from boxing_motion_ai.services.artifacts import ServiceError
from boxing_motion_ai.services.contracts import QueryResult
from boxing_motion_ai.services.interval_events import SCOPE, EventDecision, IntervalEvents
from boxing_motion_ai.services.interval_review import IntervalDecision, IntervalWorkspace, Pilot
from boxing_motion_ai.services.review_workspace import Decision, ReviewWorkspace

from .clips import ClipService
from .media import FrameService

STATUS = {"available": 200, "incomplete": 200, "missing": 404, "invalid": 422, "stale": 409}


def reply(result):
    return JSONResponse(result.model_dump(mode="json"), status_code=STATUS[result.status])


def create_app(root: Path):
    service = DashboardService(root)
    frames = FrameService(service)
    workspace = ReviewWorkspace(service)
    intervals = {p: IntervalWorkspace(service, frames, p) for p in ("original", "motion-v2")}
    clips = {p: ClipService(w, frames) for p, w in intervals.items()}
    app = FastAPI(
        title="Boxing Motion AI — local review", version="0.2.0", docs_url=None, redoc_url=None
    )
    app.state.service, app.state.frames = service, frames
    app.state.workspace = workspace
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        allowed = {
            f"http://{host}:{port}"
            for host in ("localhost", "127.0.0.1")
            for port in (5174, 8000, 4174)
        }
        if (origin and origin not in allowed) or request.headers.get(
            "sec-fetch-site"
        ) == "cross-site":
            return JSONResponse({"detail": "Only local dashboard requests are allowed"}, 403)
        if request.method not in ("GET", "HEAD", "OPTIONS") and (
            origin not in allowed
            or request.headers.get("x-review-client") != "dashboard-v0.2"
            or request.headers.get("content-type", "").split(";")[0] != "application/json"
        ):
            return JSONResponse({"detail": "Local JSON review request required"}, 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return reply(QueryResult(status=exc.status, issues=[exc.code]))

    @app.get("/api/health")
    def health():
        return {"version": "0.2.0", "mode": "local_interactive_review", "v9": "reserved"}

    @app.get("/api/inventory")
    def inventory():
        return reply(service.inventory())

    @app.get("/api/statistics")
    def statistics():
        return {v: service.statistics(v).model_dump(mode="json") for v in RECORDINGS}

    @app.get("/api/reviews")
    def reviews():
        current = workspace.snapshot()
        status = current.review_status()
        if status.status != "available":
            return reply(status)
        queue = current.review_queue()
        priorities = {r["id"]: r["priority"] for r in (queue.data or {}).get("entries", [])}
        entries = []
        for key in status.data["case_ids"]:
            v, row = key.split(":")
            result = current.review(v, int(row))
            entries.append(
                dict(
                    id=key,
                    priority=priorities.get(key, "diagnostic"),
                    review=result.model_dump(mode="json"),
                )
            )
        entries.sort(key=lambda e: (e["priority"] != "dispute", e["priority"] != "anchor", e["id"]))
        return reply(
            QueryResult(status="available", data=dict(entries=entries, summary=status.data))
        )

    def interval_reply(operation):
        try:
            return reply(QueryResult(status="available", data=operation()))
        except ServiceError:
            raise
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            raise ServiceError("invalid", "interval_validation_or_storage_failed") from exc

    @app.get("/api/intervals")
    def interval_queue(pilot: Pilot = "original"):
        return interval_reply(intervals[pilot].snapshot)

    @app.get("/api/interval-event-status")
    def interval_event_status():
        def statuses():
            events = IntervalEvents(intervals["original"])
            return {
                key: {
                    k: v
                    for k, v in events.detail(key).items()
                    if k in ("complete", "uncertain_types", "events")
                }
                for key in SCOPE
            }

        return interval_reply(statuses)

    @app.get("/api/intervals/{key}/events")
    def interval_events(key: str):
        return interval_reply(lambda: IntervalEvents(intervals["original"]).detail(key))

    @app.post("/api/intervals/{key}/events")
    def interval_event_save(key: str, decision: EventDecision):
        return interval_reply(lambda: IntervalEvents(intervals["original"]).save(key, decision))

    @app.get("/api/intervals/{key}/playback")
    def interval_playback(key: str, pilot: Pilot = "original"):
        return interval_reply(lambda: clips[pilot].prepare(key))

    @app.get("/api/intervals/{key}/clip")
    def interval_clip(key: str, request: Request, pilot: Pilot = "original"):
        try:
            return clips[pilot].response(key, request.headers.get("range"))
        except ServiceError:
            raise
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            raise ServiceError("invalid", "preview_unavailable_or_invalid_source") from exc

    @app.get("/api/intervals/{key}")
    def interval_detail(key: str, pilot: Pilot = "original"):
        return interval_reply(lambda: intervals[pilot].detail(key))

    @app.post("/api/intervals/{key}")
    def interval_save(key: str, decision: IntervalDecision, pilot: Pilot = "original"):
        return interval_reply(lambda: intervals[pilot].save(key, decision))

    @app.get("/api/curation")
    def curation():
        return reply(workspace.snapshot().curation_status())

    @app.get("/api/recordings/{recording}/annotations")
    def annotations(
        recording: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=100)
    ):
        return reply(service.annotations(recording, offset=offset, limit=limit))

    @app.get("/api/cases/{recording}/{row}")
    def case(recording: str, row: int):
        service._recording(recording)  # Reject reserved IDs before review storage access.
        current = workspace.snapshot()
        annotation = service.annotation(recording, row)
        if annotation.status != "available":
            return reply(annotation)
        return reply(
            QueryResult(
                status="available",
                data=dict(
                    annotation=annotation.data["source"],
                    labels=list(LABELS),
                    review=current.review(recording, row).model_dump(mode="json"),
                    edit=current.edit_state(recording, row).model_dump(mode="json"),
                    evidence=service.review_evidence(recording, row).model_dump(mode="json"),
                    recording=service.recording(recording).model_dump(mode="json"),
                ),
            )
        )

    @app.post("/api/reviews/{recording}/{row}")
    def save_review(recording: str, row: int, decision: Decision):
        try:
            receipt = workspace.save(recording, row, decision, frames)
        except ServiceError:
            raise
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ServiceError("invalid", "review_validation_or_storage_failed") from exc
        return reply(QueryResult(status="available", data=receipt))

    @app.get("/api/experiments")
    def experiments():
        return reply(service.experiments())

    @app.get("/api/experiments/{experiment}")
    def experiment(experiment: str):
        return reply(service.experiment(experiment))

    @app.get("/api/experiments/{experiment}/metrics/{metric}")
    def metrics(experiment: str, metric: str):
        return reply(service.model_metrics(experiment, metric))

    def get_frame(recording, index):
        try:
            return frames.frame(recording, index)
        except ServiceError:
            raise
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ServiceError("invalid", "frame_unavailable_or_invalid_source") from exc

    @app.get("/api/recordings/{recording}/frames/{index}")
    def frame(recording: str, index: int):
        metadata, _ = get_frame(recording, index)
        return reply(QueryResult(status="available", data=metadata))

    @app.get("/api/recordings/{recording}/frames/{index}/image")
    def image(recording: str, index: int):
        metadata, pixels = get_frame(recording, index)
        return Response(
            pixels,
            media_type="image/jpeg",
            headers={
                "X-Native-Frame": str(metadata["frame_index"]),
                "X-Exact-Timestamp": metadata["exact_timestamp"],
            },
        )

    # Only the built application, never the repository, datasets or reports.
    dist = Path(root) / "web" / "dist"
    if dist.is_dir() and not dist.is_symlink():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")
    return app
