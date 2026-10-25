"""The Lightning API key goes only back to the API, and only encrypted."""
import pytest

from src.cloud.framework.client import may_send_key

BASE = "https://lightning.ai/v1"


@pytest.mark.parametrize("url, sends", [
    ("https://lightning.ai/v1/artifacts/1.tar.gz", True),
    ("https://LIGHTNING.ai:443/v1/a", True),          # same origin, spelled out
    ("http://lightning.ai/v1/a", False),              # plain http: readable on the wire
    ("https://lightning.ai:8443/v1/a", False),        # another port, another server
    ("https://cdn.lightning.ai/a", False),            # another host
    ("https://s3.amazonaws.com/bucket/a?sig=x", False),
    ("https://lightning.ai.evil.example/a", False),
    ("https://user@evil.example/lightning.ai", False),
    ("/v1/relative", False),
])
def test_key_goes_only_to_the_api_over_https(url, sends):
    assert may_send_key(url, BASE) is sends


def test_an_http_base_never_gets_the_key():
    assert not may_send_key("http://localhost:8000/a", "http://localhost:8000")
