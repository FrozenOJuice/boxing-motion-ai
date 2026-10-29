"""Local upload routes. No user-supplied filesystem path, dataset ID or model selection."""

from urllib.parse import unquote

from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..analysis.common import MAX_BYTES
from ..analysis.jobs import AnalysisError, AnalysisJobs


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start_frame: int = Field(ge=0)
    stop_frame_exclusive: int = Field(ge=1)


def register(app, root):
    jobs = AnalysisJobs(root)
    app.state.analysis = jobs

    @app.exception_handler(AnalysisError)
    async def analysis_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.get("/api/analyze/videos")
    def videos():
        return {"videos": jobs.listing()}

    @app.post("/api/analyze/uploads", status_code=202)
    async def upload(request: Request):
        filename = unquote(request.headers.get("x-upload-name", "video.mp4"))
        declared = request.headers.get("content-length")
        if declared is None or not declared.isdigit() or not 0 < int(declared) <= MAX_BYTES:
            raise AnalysisError("Use a nonempty video of at most 150 MiB.", 413)
        key, directory = jobs.reserve(filename)
        total = 0
        try:
            with (directory / "source.video").open("xb") as stream:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise AnalysisError("The upload exceeds 150 MiB.", 413)
                    stream.write(chunk)
            if total != int(declared):
                raise AnalysisError("The upload was incomplete. Try again.")
            jobs.submit(key, total)
        except BaseException as exc:
            jobs.abandon(key)
            if isinstance(exc, OSError):
                raise AnalysisError(
                    "Could not save the video. Check local disk space and try again."
                ) from exc
            raise
        return {"id": key, "status": "queued"}

    @app.get("/api/analyze/videos/{key}")
    def status(key: str):
        return jobs.status(key)

    @app.get("/api/analyze/videos/{key}/preview")
    def preview(key: str):
        return FileResponse(jobs.asset(key), media_type="video/mp4")

    @app.get("/api/analyze/videos/{key}/frames/{index}")
    def frame(key: str, index: int):
        return FileResponse(jobs.asset(key, index), media_type="image/png")

    @app.get("/api/analyze/videos/{key}/activity")
    def activity(key: str):
        return jobs.activity(key)

    @app.post("/api/analyze/videos/{key}/scan")
    def scan(key: str):
        value = jobs.scan(key)
        return JSONResponse(value, status_code=202 if value["status"] == "processing" else 200)

    @app.post("/api/analyze/videos/{key}/classify")
    def classify(key: str, selection: Selection):
        return jobs.classify(key, selection.start_frame, selection.stop_frame_exclusive)

    @app.post("/api/analyze/videos/{key}/remove")
    def remove(key: str):
        jobs.remove(key)
        return {"removed": True}
