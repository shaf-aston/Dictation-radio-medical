"""Lightning AI cloud-training integration.

This package is the *only* part of the app that talks to the network. It is
strictly optional: the local dictation pipeline never imports from here, so the
app runs fully offline when cloud features are disabled. The flow is:

    stage corrections (src.training) → de-identify (privacy) → upload (uploader)
    → train on Lightning AI → download CT2 model → register (model_registry)
    → activate into the local Transcriber.
"""
