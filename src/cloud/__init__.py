"""Lightning AI cloud-training integration.

This package is the *only* part of the app that talks to the network. It is
strictly optional: the local dictation pipeline never imports from here, so the
app runs fully offline when cloud features are disabled.

Layout:
  * :mod:`src.cloud.framework` — task-agnostic core (client, registry, sync,
    job monitor).
  * :mod:`src.cloud.tasks` — one plug-in per model type (voice, text, scan) that
    supplies the parts that differ: archive layout and Lightning job spec.
  * :mod:`src.cloud.exceptions` — the :class:`CloudError` hierarchy.

Flow: stage corrections (``src.training``) → de-identify (``src.medical.deid``) → a task
builds a batch → upload + train on Lightning AI → download artifact → register
(``framework.registry``) → activate into the local model that consumes it.
"""
