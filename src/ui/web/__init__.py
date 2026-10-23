"""The FastAPI web front-end, one router per job; ``src/ui/web_app.py`` assembles them.

``state`` holds what lives between requests; ``schemas`` the request shapes;
``pages`` the page and its first-load state; ``preferences``, ``report``,
``terms``, ``diagnostics`` and ``dictate`` the endpoints, by what they are for.
"""
