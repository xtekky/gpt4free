"""
This module provides scipy versions of high_pass_filter, and low_pass_filter
as well as an additional band_pass_filter.

Of course, you will need to install scipy for these to work.

When this module is imported the high and low pass filters from this module
will be used when calling audio_segment.high_pass_filter() and
audio_segment.high_pass_filter() instead of the slower, less powerful versions
provided by pydub.effects.
"""
from scipy.signal import butter, sosfilt
from .utils import (register_pydub_effect,stereo_to_ms,ms_to_stereo)


def _mk_butter_filter(freq, type, order):
    """
    Args:
        freq: The cutoff frequency for highpass and lowpass filters. For
            band filters, a list of [low_cutoff, high_cutoff]
        type: "lowpass", "highpass", or "band"
        order: nth order butterworth filter (default: 5th order). The
            attenuation is -6dB/octave beyond the cutoff frequency (for 1st
            order). A Higher order filter will have more attenuation, each level
            adding an additional -6dB (so a 3rd order butterworth filter would
            be -18dB/octave).

    Returns:
        function which can filter a mono audio segment

    """
    def filter_fn(seg):
        assert seg.channels == 1

        nyq = 0.5 * seg.frame_rate
        try:
            freqs = [f / nyq for f in freq]
        except TypeError:
            freqs = freq / nyq

        sos = butter(order, freqs, btype=type, output='sos')
        y = sosfilt(sos, seg.get_array_of_samples())

        return seg._spawn(y.astype(seg.array_type))

    return filter_fn


@register_pydub_effect
def band_pass_filter(seg, low_cutoff_freq, high_cutoff_freq, order=5):
    filter_fn = _mk_butter_filter([low_cutoff_freq, high_cutoff_freq], 'band', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)


@register_pydub_effect
def high_pass_filter(seg, cutoff_freq, order=5):
    filter_fn = _mk_butter_filter(cutoff_freq, 'highpass', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)


@register_pydub_effect
def low_pass_filter(seg, cutoff_freq, order=5):
    filter_fn = _mk_butter_filter(cutoff_freq, 'lowpass', order=order)
    return seg.apply_mono_filter_to_each_channel(filter_fn)


@register_pydub_effect
def _eq(seg, focus_freq, bandwidth=100, mode="peak", gain_dB=0, order=2):
    """
    Args:
        focus_freq - middle frequency or known frequency of band (in Hz)
        bandwidth - range of the equalizer band
        mode - Mode of Equalization(Peak/Notch(Bell Curve),High Shelf, Low Shelf)
        order - Rolloff factor(1 - 6dB/Octave 2 - 12dB/Octave)
    
    Returns:
        Equalized/Filtered AudioSegment
    """
    filt_mode = ["peak", "low_shelf", "high_shelf"]
    if mode not in filt_mode:
        raise ValueError("Incorrect Mode Selection")
        
    if gain_dB >= 0:
        if mode == "peak":
            sec = band_pass_filter(seg, focus_freq - bandwidth/2, focus_freq + bandwidth/2, order = order)
            seg = seg.overlay(sec - (3 - gain_dB))
            return seg
        
        if mode == "low_shelf":
            sec = low_pass_filter(seg, focus_freq, order=order)
            seg = seg.overlay(sec - (3 - gain_dB))
            return seg
        
        if mode == "high_shelf":
            sec = high_pass_filter(seg, focus_freq, order=order)
            seg = seg.overlay(sec - (3 - gain_dB))
            return seg
        
    if gain_dB < 0:
        if mode == "peak":
            sec = high_pass_filter(seg, focus_freq - bandwidth/2, order=order)
            seg = seg.overlay(sec - (3 + gain_dB)) + gain_dB
            sec = low_pass_filter(seg, focus_freq + bandwidth/2, order=order)
            seg = seg.overlay(sec - (3 + gain_dB)) + gain_dB
            return seg
        
        if mode == "low_shelf":
            sec = high_pass_filter(seg, focus_freq, order=order)
            seg = seg.overlay(sec - (3 + gain_dB)) + gain_dB
            return seg
        
        if mode=="high_shelf":
            sec=low_pass_filter(seg, focus_freq, order=order)
            seg=seg.overlay(sec - (3 + gain_dB)) +gain_dB
            return seg
        

@register_pydub_effect
def eq(seg, focus_freq, bandwidth=100, channel_mode="L+R", filter_mode="peak", gain_dB=0, order=2):
    """
    Args:
        focus_freq - middle frequency or known frequency of band (in Hz)
        bandwidth - range of the equalizer band
        channel_mode - Select Channels to be affected by the filter.
            L+R - Standard Stereo Filter
            L - Only Left Channel is Filtered
            R - Only Right Channel is Filtered
            M+S - Blumlien Stereo Filter(Mid-Side)
            M - Only Mid Channel is Filtered
            S - Only Side Channel is Filtered
            Mono Audio Segments are completely filtered.
        filter_mode - Mode of Equalization(Peak/Notch(Bell Curve),High Shelf, Low Shelf)
        order - Rolloff factor(1 - 6dB/Octave 2 - 12dB/Octave)
    
    Returns:
        Equalized/Filtered AudioSegment
    """
    channel_modes = ["L+R", "M+S", "L", "R", "M", "S"]
    if channel_mode not in channel_modes:
        raise ValueError("Incorrect Channel Mode Selection")
        
    if seg.channels == 1:
        return _eq(seg, focus_freq, bandwidth, filter_mode, gain_dB, order)
        
    if channel_mode == "L+R":
        return _eq(seg, focus_freq, bandwidth, filter_mode, gain_dB, order)
        
    if channel_mode == "L":
        seg = seg.split_to_mono()
        seg = [_eq(seg[0], focus_freq, bandwidth, filter_mode, gain_dB, order), seg[1]]
        return AudioSegment.from_mono_audio_segements(seg[0], seg[1])
        
    if channel_mode == "R":
        seg = seg.split_to_mono()
        seg = [seg[0], _eq(seg[1], focus_freq, bandwidth, filter_mode, gain_dB, order)]
        return AudioSegment.from_mono_audio_segements(seg[0], seg[1])
        
    if channel_mode == "M+S":
        seg = stereo_to_ms(seg)
        seg = _eq(seg, focus_freq, bandwidth, filter_mode, gain_dB, order)
        return ms_to_stereo(seg)
        
    if channel_mode == "M":
        seg = stereo_to_ms(seg).split_to_mono()
        seg = [_eq(seg[0], focus_freq, bandwidth, filter_mode, gain_dB, order), seg[1]]
        seg = AudioSegment.from_mono_audio_segements(seg[0], seg[1])
        return ms_to_stereo(seg)
        
    if channel_mode == "S":
        seg = stereo_to_ms(seg).split_to_mono()
        seg = [seg[0], _eq(seg[1], focus_freq, bandwidth, filter_mode, gain_dB, order)]
        seg = AudioSegment.from_mono_audio_segements(seg[0], seg[1])
        return ms_to_stereo(seg)


