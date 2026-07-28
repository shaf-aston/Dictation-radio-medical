"""The shared JSON read/write helpers every settings and registry file goes through.

`write_json` exists to make a write atomic, so a crash or a collision can never
leave a truncated file that reads back as the empty default. That promise is only
worth anything under the condition it was written for — more than one writer —
because the desktop app and the web app run against the same
`dictation_settings.json`. So the load-bearing test here is the concurrent one.
"""

from __future__ import annotations

import json
import threading

from src.core.json_store import append_jsonl, read_json, read_jsonl, write_json


class TestRoundTrip:
    def test_write_then_read(self, tmp_path) -> None:
        path = tmp_path / "thing.json"
        write_json(path, {"model": "small.en", "beam": 5})
        assert read_json(path, {}) == {"model": "small.en", "beam": 5}

    def test_parent_directories_are_created(self, tmp_path) -> None:
        path = tmp_path / "deep" / "deeper" / "thing.json"
        write_json(path, [1, 2, 3])
        assert read_json(path, None) == [1, 2, 3]

    def test_a_missing_file_reads_as_the_default(self, tmp_path) -> None:
        assert read_json(tmp_path / "absent.json", {"fallback": True}) == {"fallback": True}

    def test_a_corrupt_file_reads_as_the_default(self, tmp_path) -> None:
        path = tmp_path / "corrupt.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert read_json(path, {"fallback": True}) == {"fallback": True}


class TestConcurrentWriters:
    """Two front-ends write these files, so a shared temp name corrupts them.

    With one `<target>.tmp` for every writer, the losing thread either finds its
    temp file already consumed or moves a half-written one over the target. Either
    way the file this function promised to protect is the file that breaks.
    """

    def test_the_file_is_never_left_corrupt(self, tmp_path) -> None:
        path = tmp_path / "settings.json"
        # Payloads big enough that a write is not one atomic OS operation, so an
        # interleaving has room to actually happen.
        payloads = [{"writer": i, "pad": ["x" * 200] * 50} for i in range(12)]
        barrier = threading.Barrier(len(payloads))

        def writer(payload: dict) -> None:
            barrier.wait()  # start together, to actually contend
            for _ in range(5):
                write_json(path, payload)

        threads = [threading.Thread(target=writer, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Whichever writer landed last, the file must be complete and be exactly
        # one of the payloads — never a blend of two, never truncated.
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded in payloads

    def test_no_temp_files_are_left_behind(self, tmp_path) -> None:
        path = tmp_path / "settings.json"
        threads = [
            threading.Thread(target=write_json, args=(path, {"writer": i}))
            for i in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
        assert not leftovers, f"stray temp files: {leftovers}"


class TestFailedWrite:
    def test_unserialisable_data_is_warned_about_not_raised(self, tmp_path) -> None:
        # A logging/settings write must never take down the operation it records.
        path = tmp_path / "bad.json"
        write_json(path, {"handle": object()})
        assert not path.exists()

    def test_a_failed_write_leaves_no_temp_file(self, tmp_path) -> None:
        write_json(tmp_path / "bad.json", {"handle": object()})
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_write_does_not_destroy_the_previous_file(self, tmp_path) -> None:
        # The old value surviving is the whole point of writing to one side first.
        path = tmp_path / "settings.json"
        write_json(path, {"model": "small.en"})
        write_json(path, {"handle": object()})
        assert read_json(path, {}) == {"model": "small.en"}


class TestJsonLines:
    def test_append_and_read_back(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        assert append_jsonl(path, [{"a": 1}, {"a": 2}]) == 2
        assert append_jsonl(path, [{"a": 3}]) == 1
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}, {"a": 3}]

    def test_a_bad_line_does_not_lose_the_good_ones(self, tmp_path) -> None:
        # An audit trail with one damaged line is still evidence; discarding the
        # whole file over it would be the worse failure.
        path = tmp_path / "audit.jsonl"
        append_jsonl(path, [{"a": 1}])
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("{truncated\n")
        append_jsonl(path, [{"a": 2}])
        assert {"a": 1} in read_jsonl(path)
        assert {"a": 2} in read_jsonl(path)
