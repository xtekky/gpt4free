"""Frame packing and the capture pipeline."""

from __future__ import annotations

import pytest

from remote_desktop.capture import (
    FRAME_HEADER_SIZE,
    FRAME_MAGIC,
    CaptureUnavailableError,
    ScreenCapture,
    pack_frame,
    unpack_frame,
)

from .fakes import FakeGrabber


def test_pack_unpack_round_trip():
    payload = b"\xff\xd8\xff\xe0jpeg-bytes"
    packed = pack_frame(7, payload)
    assert packed[:1] == FRAME_MAGIC
    assert len(packed) == FRAME_HEADER_SIZE + len(payload)
    sequence, data = unpack_frame(packed)
    assert sequence == 7
    assert data == payload


def test_pack_frame_wraps_sequence_to_32_bits():
    sequence, _ = unpack_frame(pack_frame(2**32 + 5, b"x"))
    assert sequence == 5


def test_unpack_frame_rejects_foreign_payloads():
    with pytest.raises(ValueError):
        unpack_frame(b"")
    with pytest.raises(ValueError):
        unpack_frame(b"\x02\x00\x00\x00\x01data")
    with pytest.raises(ValueError):
        unpack_frame(b"\x01\x00\x00")


def test_screen_capture_downscales_and_encodes(monkeypatch):
    grabber = FakeGrabber(width=800, height=600)
    monkeypatch.setattr("remote_desktop.capture.mss.MSS", lambda **kwargs: grabber)

    capture = ScreenCapture(monitor=1, max_width=400, quality=50)
    assert capture.screen_size() == (800, 600)

    frame = capture.grab()
    assert frame.width == 400
    assert frame.height == 300
    assert frame.data[:2] == b"\xff\xd8"
    assert grabber.grabs == 1

    capture.close()
    assert grabber.closed is True


def test_screen_capture_keeps_small_frames_untouched(monkeypatch):
    grabber = FakeGrabber(width=320, height=240)
    monkeypatch.setattr("remote_desktop.capture.mss.MSS", lambda **kwargs: grabber)

    frame = ScreenCapture(monitor=1, max_width=1280).grab()
    assert (frame.width, frame.height) == (320, 240)


def test_screen_capture_rejects_unknown_monitor(monkeypatch):
    monkeypatch.setattr(
        "remote_desktop.capture.mss.MSS", lambda **kwargs: FakeGrabber()
    )
    capture = ScreenCapture(monitor=9)
    with pytest.raises(CaptureUnavailableError):
        capture.grab()


def test_screen_capture_falls_back_when_cursor_unsupported(monkeypatch):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        if "with_cursor" in kwargs:
            raise TypeError("unexpected keyword argument 'with_cursor'")
        return FakeGrabber()

    monkeypatch.setattr("remote_desktop.capture.mss.MSS", factory)
    capture = ScreenCapture(monitor=1)
    assert capture.grab().width == 800
    assert calls == [{"with_cursor": True}, {}]
