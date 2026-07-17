"""Lightning AI REST client — task-agnostic.

Wraps the Lightning AI HTTP API for the four operations every training task
needs: upload a batch archive, submit a job, poll job status, and download the
resulting artifact. The REST surface (rather than the heavier SDK or SSH) keeps
this dependency-light and firewall-friendly — only ``httpx`` is required, which
the project already ships.

Credentials: the API key is read from the OS keychain via :mod:`keyring` and is
never written to ``dictation_settings.json`` or any log. The project id (not a
secret) lives in settings.

The job payload is generic: a :class:`~src.cloud.tasks.base.JobSpec` (entrypoint
+ compute + args) is supplied by the task, so adding a new model type never
touches this file. Lightning AI's exact REST paths evolve; they are centralised
as constants here so a future API change is a one-file edit.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.cloud.exceptions import AuthError, CloudError, QuotaError
from src.core.keychain import clear_secret, get_secret, store_secret

if TYPE_CHECKING:
    from src.cloud.tasks.base import JobSpec

logger = logging.getLogger(__name__)

_KEYRING_KEY = "lightning_api_key"
_BASE_URL = "https://lightning.ai/api/v1"
_TIMEOUT = 60.0


def store_api_key(api_key: str) -> None:
    """Persist the Lightning AI API key in the OS keychain."""
    store_secret(_KEYRING_KEY, api_key)
    logger.info("Lightning AI API key stored in OS keychain")


def get_api_key() -> Optional[str]:
    """Retrieve the API key from the OS keychain, or None if unset."""
    return get_secret(_KEYRING_KEY)


def clear_api_key() -> None:
    """Remove the stored API key from the OS keychain."""
    clear_secret(_KEYRING_KEY)


class LightningAIClient:
    """Minimal REST wrapper for Lightning AI training jobs."""

    def __init__(self, project_id: str, api_key: Optional[str] = None,
                 base_url: str = _BASE_URL) -> None:
        self.project_id = project_id
        self._api_key = api_key or get_api_key()
        self._base_url = base_url.rstrip("/")
        if not self._api_key:
            raise AuthError("No Lightning AI API key configured.")
        import httpx
        self._http = httpx.Client(timeout=_TIMEOUT, headers=self._headers())

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs):
        import httpx
        url = f"{self._base_url}{path}"
        try:
            resp = self._http.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise CloudError(f"Network error calling {path}: {exc}") from exc
        if resp.status_code in (401, 403):
            raise AuthError(f"Authentication rejected by Lightning AI ({resp.status_code}).")
        if resp.status_code == 429:
            raise QuotaError("Lightning AI rate limit / quota exceeded.")
        if resp.status_code >= 400:
            raise CloudError(f"Lightning AI error {resp.status_code}: {resp.text[:200]}")
        return resp

    def check_connectivity(self) -> bool:
        """Lightweight reachability + auth probe."""
        try:
            self._request("GET", f"/projects/{self.project_id}")
            return True
        except CloudError as exc:
            logger.warning("Connectivity check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Data upload
    # ------------------------------------------------------------------

    def upload_training_data(self, archive_path: Path) -> str:
        """Upload a batch archive to Lightning storage; return its cloud URL."""
        archive_path = Path(archive_path)
        with open(archive_path, "rb") as fh:
            files = {"file": (archive_path.name, fh, "application/gzip")}
            resp = self._request(
                "POST",
                f"/projects/{self.project_id}/storage/upload",
                files=files,
            )
        data = resp.json()
        url = data.get("url") or data.get("path")
        if not url:
            raise CloudError("Upload succeeded but no storage URL returned.")
        logger.info("Uploaded %s → %s", archive_path.name, url)
        return url

    # ------------------------------------------------------------------
    # Training jobs
    # ------------------------------------------------------------------

    def submit_training_job(self, spec: "JobSpec") -> str:
        """Submit a fine-tuning job described by *spec*; return the job id.

        The task owns *spec* (name, entrypoint, compute, args), so this method
        is identical for voice, text-correction, and scan models.
        """
        payload = {
            "name": spec.name,
            "entrypoint": spec.entrypoint,
            "compute": spec.compute,
            "args": spec.args,
        }
        resp = self._request(
            "POST", f"/projects/{self.project_id}/jobs", json=payload
        )
        job_id = resp.json().get("id") or resp.json().get("job_id")
        if not job_id:
            raise CloudError("Job submission returned no job id.")
        logger.info("Submitted job %s (%s)", job_id, spec.name)
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        """Return ``{status, progress, error, artifact_url}`` for a job.

        ``status`` is normalised to one of: queued, running, completed, failed.
        """
        resp = self._request("GET", f"/projects/{self.project_id}/jobs/{job_id}")
        data = resp.json()
        raw = (data.get("status") or "").lower()
        status = {
            "pending": "queued", "queued": "queued",
            "running": "running", "in_progress": "running",
            "completed": "completed", "succeeded": "completed", "success": "completed",
            "failed": "failed", "error": "failed", "cancelled": "failed",
        }.get(raw, raw or "queued")
        return {
            "status": status,
            "progress": data.get("progress"),
            "error": data.get("error"),
            "artifact_url": data.get("artifact_url") or data.get("artifacts_url"),
        }

    # ------------------------------------------------------------------
    # Artefact download
    # ------------------------------------------------------------------

    def download_artifact(self, artifact_url: str, dest_dir: Path) -> Path:
        """Download a model artefact archive and return the local archive path."""
        import httpx
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "model_artifact.tar.gz"
        # Bounded read timeout: a stalled download must not hang the poll thread
        # forever. The connect timeout is short; reads get a generous 5 min/chunk.
        timeout = httpx.Timeout(_TIMEOUT, read=300.0)
        # Only attach the Lightning auth header when the artifact is served from
        # the API host. Lightning typically returns a pre-signed CDN/S3 URL on a
        # different host; sending the Bearer key there would leak it off-device.
        from urllib.parse import urlparse
        same_host = urlparse(artifact_url).netloc == urlparse(self._base_url).netloc
        headers = self._headers() if same_host else {}
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("GET", artifact_url, headers=headers) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
        except httpx.HTTPError as exc:
            raise CloudError(f"Artifact download failed: {exc}") from exc
        logger.info("Downloaded artifact → %s", dest)
        return dest
