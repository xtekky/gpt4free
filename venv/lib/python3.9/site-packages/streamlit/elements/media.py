# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import re
from typing import TYPE_CHECKING, Optional, Tuple, Union, cast

from typing_extensions import Final, TypeAlias
from validators import url

import streamlit as st
from streamlit import runtime, type_util
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Audio_pb2 import Audio as AudioProto
from streamlit.proto.Video_pb2 import Video as VideoProto
from streamlit.runtime import caching
from streamlit.runtime.metrics_util import gather_metrics

if TYPE_CHECKING:
    from typing import Any

    from numpy import typing as npt

    from streamlit.delta_generator import DeltaGenerator

MediaData: TypeAlias = Union[
    str, bytes, io.BytesIO, io.RawIOBase, io.BufferedReader, "npt.NDArray[Any]", None
]


class MediaMixin:
    @gather_metrics("audio")
    def audio(
        self,
        data: MediaData,
        format: str = "audio/wav",
        start_time: int = 0,
        *,
        sample_rate: Optional[int] = None,
    ) -> "DeltaGenerator":
        """Display an audio player.

        Parameters
        ----------
        data : str, bytes, BytesIO, numpy.ndarray, or file opened with
                io.open().
            Raw audio data, filename, or a URL pointing to the file to load.
            Raw data formats must include all necessary file headers to match the file
            format specified via ``format``.
            If ``data`` is a numpy array, it must either be a 1D array of the waveform
            or a 2D array of shape ``(num_channels, num_samples)`` with waveforms
            for all channels. See the default channel order at
            http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
        format : str
            The mime type for the audio file. Defaults to 'audio/wav'.
            See https://tools.ietf.org/html/rfc4281 for more info.
        start_time: int
            The time from which this element should start playing.
        sample_rate: int or None
            The sample rate of the audio data in samples per second. Only required if
            ``data`` is a numpy array.

        Example
        -------
        >>> import streamlit as st
        >>> import numpy as np
        >>>
        >>> audio_file = open('myaudio.ogg', 'rb')
        >>> audio_bytes = audio_file.read()
        >>>
        >>> st.audio(audio_bytes, format='audio/ogg')
        >>>
        >>> sample_rate = 44100  # 44100 samples per second
        >>> seconds = 2  # Note duration of 2 seconds
        >>> frequency_la = 440  # Our played note will be 440 Hz
        >>> # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
        >>> t = np.linspace(0, seconds, seconds * sample_rate, False)
        >>> # Generate a 440 Hz sine wave
        >>> note_la = np.sin(frequency_la * t * 2 * np.pi)
        >>>
        >>> st.audio(note_la, sample_rate=sample_rate)

        .. output::
           https://doc-audio.streamlitapp.com/
           height: 865px

        """
        audio_proto = AudioProto()
        coordinates = self.dg._get_delta_path_str()

        is_data_numpy_array = type_util.is_type(data, "numpy.ndarray")

        if is_data_numpy_array and sample_rate is None:
            raise StreamlitAPIException(
                "`sample_rate` must be specified when `data` is a numpy array."
            )
        if not is_data_numpy_array and sample_rate is not None:
            st.warning(
                "Warning: `sample_rate` will be ignored since data is not a numpy "
                "array."
            )

        marshall_audio(coordinates, audio_proto, data, format, start_time, sample_rate)
        return self.dg._enqueue("audio", audio_proto)

    @gather_metrics("video")
    def video(
        self,
        data: MediaData,
        format: str = "video/mp4",
        start_time: int = 0,
    ) -> "DeltaGenerator":
        """Display a video player.

        Parameters
        ----------
        data : str, bytes, BytesIO, numpy.ndarray, or file opened with
                io.open().
            Raw video data, filename, or URL pointing to a video to load.
            Includes support for YouTube URLs.
            Numpy arrays and raw data formats must include all necessary file
            headers to match specified file format.
        format : str
            The mime type for the video file. Defaults to 'video/mp4'.
            See https://tools.ietf.org/html/rfc4281 for more info.
        start_time: int
            The time from which this element should start playing.

        Example
        -------
        >>> import streamlit as st
        >>>
        >>> video_file = open('myvideo.mp4', 'rb')
        >>> video_bytes = video_file.read()
        >>>
        >>> st.video(video_bytes)

        .. output::
           https://doc-video.streamlitapp.com/
           height: 700px

        .. note::
           Some videos may not display if they are encoded using MP4V (which is an export option in OpenCV), as this codec is
           not widely supported by browsers. Converting your video to H.264 will allow the video to be displayed in Streamlit.
           See this `StackOverflow post <https://stackoverflow.com/a/49535220/2394542>`_ or this
           `Streamlit forum post <https://discuss.streamlit.io/t/st-video-doesnt-show-opencv-generated-mp4/3193/2>`_
           for more information.

        """
        video_proto = VideoProto()
        coordinates = self.dg._get_delta_path_str()
        marshall_video(coordinates, video_proto, data, format, start_time)
        return self.dg._enqueue("video", video_proto)

    @property
    def dg(self) -> "DeltaGenerator":
        """Get our DeltaGenerator."""
        return cast("DeltaGenerator", self)


# Regular expression explained at https://regexr.com/4n2l2 Covers any youtube
# URL (incl. shortlinks and embed links) and extracts its code.
YOUTUBE_RE: Final = re.compile(
    # Protocol
    r"http(?:s?):\/\/"
    # Domain
    r"(?:www\.)?youtu(?:be\.com|\.be)\/"
    # Path and query string
    r"(?P<watch>(watch\?v=)|embed\/)?(?P<code>[\w\-\_]*)(&(amp;)?[\w\?=]*)?"
)


def _reshape_youtube_url(url: str) -> Optional[str]:
    """Return whether URL is any kind of YouTube embed or watch link.  If so,
    reshape URL into an embed link suitable for use in an iframe.

    If not a YouTube URL, return None.

    Parameters
    ----------
        url : str

    Example
    -------
    >>> print(_reshape_youtube_url('https://youtu.be/_T8LGqJtuGc'))

    .. output::
        https://www.youtube.com/embed/_T8LGqJtuGc
    """
    match = YOUTUBE_RE.match(url)
    if match:
        return "https://www.youtube.com/embed/{code}".format(**match.groupdict())
    return None


def _marshall_av_media(
    coordinates: str,
    proto: Union[AudioProto, VideoProto],
    data: MediaData,
    mimetype: str,
) -> None:
    """Fill audio or video proto based on contents of data.

    Given a string, check if it's a url; if so, send it out without modification.
    Otherwise assume strings are filenames and let any OS errors raise.

    Load data either from file or through bytes-processing methods into a
    MediaFile object.  Pack proto with generated Tornado-based URL.

    (When running in "raw" mode, we won't actually load data into the
    MediaFileManager, and we'll return an empty URL.)
    """
    # Audio and Video methods have already checked if this is a URL by this point.

    if data is None:
        # Allow empty values so media players can be shown without media.
        return

    data_or_filename: Union[bytes, str]
    if isinstance(data, (str, bytes)):
        # Pass strings and bytes through unchanged
        data_or_filename = data
    elif isinstance(data, io.BytesIO):
        data.seek(0)
        data_or_filename = data.getvalue()
    elif isinstance(data, io.RawIOBase) or isinstance(data, io.BufferedReader):
        data.seek(0)
        read_data = data.read()
        if read_data is None:
            return
        else:
            data_or_filename = read_data
    elif type_util.is_type(data, "numpy.ndarray"):
        data_or_filename = data.tobytes()
    else:
        raise RuntimeError("Invalid binary data format: %s" % type(data))

    if runtime.exists():
        file_url = runtime.get_instance().media_file_mgr.add(
            data_or_filename, mimetype, coordinates
        )
        caching.save_media_data(data_or_filename, mimetype, coordinates)
    else:
        # When running in "raw mode", we can't access the MediaFileManager.
        file_url = ""

    proto.url = file_url


def marshall_video(
    coordinates: str,
    proto: VideoProto,
    data: MediaData,
    mimetype: str = "video/mp4",
    start_time: int = 0,
) -> None:
    """Marshalls a video proto, using url processors as needed.

    Parameters
    ----------
    coordinates : str
    proto : the proto to fill. Must have a string field called "data".
    data : str, bytes, BytesIO, numpy.ndarray, or file opened with
           io.open().
        Raw video data or a string with a URL pointing to the video
        to load. Includes support for YouTube URLs.
        If passing the raw data, this must include headers and any other
        bytes required in the actual file.
    mimetype : str
        The mime type for the video file. Defaults to 'video/mp4'.
        See https://tools.ietf.org/html/rfc4281 for more info.
    start_time : int
        The time from which this element should start playing. (default: 0)
    """

    proto.start_time = start_time

    # "type" distinguishes between YouTube and non-YouTube links
    proto.type = VideoProto.Type.NATIVE

    if isinstance(data, str) and url(data):
        youtube_url = _reshape_youtube_url(data)
        if youtube_url:
            proto.url = youtube_url
            proto.type = VideoProto.Type.YOUTUBE_IFRAME
        else:
            proto.url = data

    else:
        _marshall_av_media(coordinates, proto, data, mimetype)


def _validate_and_normalize(data: "npt.NDArray[Any]") -> Tuple[bytes, int]:
    """Validates and normalizes numpy array data.
    We validate numpy array shape (should be 1d or 2d)
    We normalize input data to int16 [-32768, 32767] range.

    Parameters
    ----------
    data : numpy array
        numpy array to be validated and normalized

    Returns
    -------
    Tuple of (bytes, int)
        (bytes, nchan)
        where
         - bytes : bytes of normalized numpy array converted to int16
         - nchan : number of channels for audio signal. 1 for mono, or 2 for stereo.
    """
    # we import numpy here locally to import it only when needed (when numpy array given
    # to st.audio data)
    import numpy as np

    data = np.array(data, dtype=float)

    if len(data.shape) == 1:
        nchan = 1
    elif len(data.shape) == 2:
        # In wave files,channels are interleaved. E.g.,
        # "L1R1L2R2..." for stereo. See
        # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
        # for channel ordering
        nchan = data.shape[0]
        data = data.T.ravel()
    else:
        raise StreamlitAPIException("Numpy array audio input must be a 1D or 2D array.")

    if data.size == 0:
        return data.astype(np.int16).tobytes(), nchan

    max_abs_value = np.max(np.abs(data))
    # 16-bit samples are stored as 2's-complement signed integers,
    # ranging from -32768 to 32767.
    # scaled_data is PCM 16 bit numpy array, that's why we multiply [-1, 1] float
    # values to 32_767 == 2 ** 15 - 1.
    np_array = (data / max_abs_value) * 32767
    scaled_data = np_array.astype(np.int16)
    return scaled_data.tobytes(), nchan


def _make_wav(data: "npt.NDArray[Any]", sample_rate: int) -> bytes:
    """
    Transform a numpy array to a PCM bytestring
    We use code from IPython display module to convert numpy array to wave bytes
    https://github.com/ipython/ipython/blob/1015c392f3d50cf4ff3e9f29beede8c1abfdcb2a/IPython/lib/display.py#L146
    """
    # we import wave here locally to import it only when needed (when numpy array given
    # to st.audio data)
    import wave

    scaled, nchan = _validate_and_normalize(data)

    with io.BytesIO() as fp, wave.open(fp, mode="wb") as waveobj:
        waveobj.setnchannels(nchan)
        waveobj.setframerate(sample_rate)
        waveobj.setsampwidth(2)
        waveobj.setcomptype("NONE", "NONE")
        waveobj.writeframes(scaled)
        return fp.getvalue()


def _maybe_convert_to_wav_bytes(
    data: MediaData, sample_rate: Optional[int]
) -> MediaData:
    """Convert data to wav bytes if the data type is numpy array."""
    if type_util.is_type(data, "numpy.ndarray") and sample_rate is not None:
        data = _make_wav(cast("npt.NDArray[Any]", data), sample_rate)
    return data


def marshall_audio(
    coordinates: str,
    proto: AudioProto,
    data: MediaData,
    mimetype: str = "audio/wav",
    start_time: int = 0,
    sample_rate: Optional[int] = None,
) -> None:
    """Marshalls an audio proto, using data and url processors as needed.

    Parameters
    ----------
    coordinates : str
    proto : The proto to fill. Must have a string field called "url".
    data : str, bytes, BytesIO, numpy.ndarray, or file opened with
            io.open()
        Raw audio data or a string with a URL pointing to the file to load.
        If passing the raw data, this must include headers and any other bytes
        required in the actual file.
    mimetype : str
        The mime type for the audio file. Defaults to "audio/wav".
        See https://tools.ietf.org/html/rfc4281 for more info.
    start_time : int
        The time from which this element should start playing. (default: 0)
    sample_rate: int or None
        Optional param to provide sample_rate in case of numpy array
    """

    proto.start_time = start_time

    if isinstance(data, str) and url(data):
        proto.url = data

    else:
        data = _maybe_convert_to_wav_bytes(data, sample_rate)
        _marshall_av_media(coordinates, proto, data, mimetype)
