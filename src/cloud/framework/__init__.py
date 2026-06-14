"""Task-agnostic Lightning AI training core.

This package holds the machinery shared by every cloud-training task — the REST
client, the job-status monitor, the model registry, and the upload→train→
download→register sync orchestrator. A *task* (see :mod:`src.cloud.tasks`)
supplies the parts that differ per model: how to build the batch archive, which
entrypoint/compute the Lightning job uses, and how the downloaded artifact is
laid out. Nothing here knows about Whisper, text correction, or scans.
"""
