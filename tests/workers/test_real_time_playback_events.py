from tts_audiobook_tool.real_time_playback_events import (
    RealTimePlaybackBuffer,
    RealTimePlaybackEvents,
    RealTimePlaybackProgress,
    RealTimePlaybackSegmentText,
)


def test_realtime_event_sink_is_scoped() -> None:
    received = []

    RealTimePlaybackEvents.emit(RealTimePlaybackBuffer(1.0))
    with RealTimePlaybackEvents.using_sink(received.append):
        RealTimePlaybackEvents.emit(RealTimePlaybackProgress(1, 3, 1))
    RealTimePlaybackEvents.emit(RealTimePlaybackBuffer(2.0))

    assert received == [RealTimePlaybackProgress(1, 3, 1)]


def test_realtime_event_sink_carries_segment_text() -> None:
    received = []
    event = RealTimePlaybackSegmentText(0, "Hello.", 0, 48000, 12000)

    with RealTimePlaybackEvents.using_sink(received.append):
        RealTimePlaybackEvents.emit(event)

    assert received == [event]
