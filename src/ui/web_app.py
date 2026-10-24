"""Radio Dictate's web front-end: ``python -m src.ui.web_app``.

Only assembly lives here: start-up, and which routers make the app. The
endpoints are in ``src/ui/web/``, one module per job, and every rule they
apply lives below them in ``features/``, ``medical/`` and ``dictation/``,
shared with the desktop.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core import event_log
from src.core.logging_setup import setup_logging
from src.core.paths import startup_cleanup
from src.dictation.asr.models import resolve_model
from src.ui.web import diagnostics, dictate, pages, preferences, report, state, terms
from src.ui.web.local_only import LocalOnlyMiddleware, allowed_hosts

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Prepare the app state on startup."""
    settings = state.settings()
    # Start the diary first, so the warm-up lines below are the first thing the
    # developer panel can show. A panel that only starts recording once you
    # open it can never explain the slow start you opened it to look at.
    event_log.configure(int(settings.get("event_log_max")))
    event_log.attach_to_logging()
    logger.info("Preparing web app state...")
    state.drop_engine()
    # Same startup housekeeping as the desktop GUI (temp files, old
    # autosaves, legacy cache locations): a web-only user must not miss it.
    startup_cleanup(settings.get("autosave_retention_days"))
    warm_up_singletons()
    yield
    logger.info("Shutting down")


def warm_up_singletons() -> None:
    """Pre-build the slow spelling index and speech models off the request path.

    Left lazy, both are built inside the *first* dictation and stall it by
    seconds. Warming at startup on a background thread makes the first real
    request fast. Best-effort: warm-up never blocks or fails app startup.
    """
    from src.dictation.warmup import postprocess_warmers, transcriber_warmer, warm_up_async

    settings = state.settings()
    model_size = resolve_model(settings.get("model_size"))
    live_model_size = resolve_model(settings.get("live_model_size"))
    warmers = postprocess_warmers() + [transcriber_warmer(model_size)]
    # The live model decodes every word shown while the radiologist is still
    # speaking: leaving it lazy means the FIRST dictation after every
    # restart pays its full model-load cost on that exact path.
    if live_model_size != model_size:
        warmers.append(transcriber_warmer(live_model_size))
    warm_up_async(warmers)


def create_app() -> FastAPI:
    application = FastAPI(title="Radio Dictate Web", lifespan=lifespan)
    for module in (pages, preferences, report, terms, diagnostics, dictate):
        application.include_router(module.router)
    application.add_middleware(LocalOnlyMiddleware, hosts=allowed_hosts(state.settings().get))
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = state.settings()
    setup_logging()
    uvicorn.run(app, host=str(settings.get("web_host")), port=int(settings.get("web_port")))
