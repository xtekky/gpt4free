"""
Each generator will return float samples from -1.0 to 1.0, which can be 
converted to actual audio with 8, 16, 24, or 32 bit depth using the
SiganlGenerator.to_audio_segment() method (on any of it's subclasses).

See Wikipedia's "waveform" page for info on some of the generators included 
here: http://en.wikipedia.org/wiki/Waveform
"""

import math
import array
import itertools
import random
from .audio_segment import AudioSegment
from .utils import (
    db_to_float,
    get_frame_width,
    get_array_type,
    get_min_max_value
)



class SignalGenerator(object):
    def __init__(self, sample_rate=44100, bit_depth=16):
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth

    def to_audio_segment(self, duration=1000.0, volume=0.0):
        """
        Duration in milliseconds
            (default: 1 second)
        Volume in DB relative to maximum amplitude
            (default 0.0 dBFS, which is the maximum value)
        """
        minval, maxval = get_min_max_value(self.bit_depth)
        sample_width = get_frame_width(self.bit_depth)
        array_type = get_array_type(self.bit_depth)

        gain = db_to_float(volume)
        sample_count = int(self.sample_rate * (duration / 1000.0))

        sample_data = (int(val * maxval * gain) for val in self.generate())
        sample_data = itertools.islice(sample_data, 0, sample_count)

        data = array.array(array_type, sample_data)
        
        try:
            data = data.tobytes()
        except:
            data = data.tostring()

        return AudioSegment(data=data, metadata={
            "channels": 1,
            "sample_width": sample_width,
            "frame_rate": self.sample_rate,
            "frame_width": sample_width,
        })

    def generate(self):
        raise NotImplementedError("SignalGenerator subclasses must implement the generate() method, and *should not* call the superclass implementation.")



class Sine(SignalGenerator):
    def __init__(self, freq, **kwargs):
        super(Sine, self).__init__(**kwargs)
        self.freq = freq

    def generate(self):
        sine_of = (self.freq * 2 * math.pi) / self.sample_rate
        sample_n = 0
        while True:
            yield math.sin(sine_of * sample_n)
            sample_n += 1



class Pulse(SignalGenerator):
    def __init__(self, freq, duty_cycle=0.5, **kwargs):
        super(Pulse, self).__init__(**kwargs)
        self.freq = freq
        self.duty_cycle = duty_cycle

    def generate(self):
        sample_n = 0

        # in samples
        cycle_length = self.sample_rate / float(self.freq)
        pulse_length = cycle_length * self.duty_cycle

        while True:
            if (sample_n % cycle_length) < pulse_length:
                yield 1.0
            else:
                yield -1.0
            sample_n += 1



class Square(Pulse):
    def __init__(self, freq, **kwargs):
        kwargs['duty_cycle'] = 0.5
        super(Square, self).__init__(freq, **kwargs)



class Sawtooth(SignalGenerator):
    def __init__(self, freq, duty_cycle=1.0, **kwargs):
        super(Sawtooth, self).__init__(**kwargs)
        self.freq = freq
        self.duty_cycle = duty_cycle

    def generate(self):
        sample_n = 0

        # in samples
        cycle_length = self.sample_rate / float(self.freq)
        midpoint = cycle_length * self.duty_cycle
        ascend_length = midpoint
        descend_length = cycle_length - ascend_length

        while True:
            cycle_position = sample_n % cycle_length
            if cycle_position < midpoint:
                yield (2 * cycle_position / ascend_length) - 1.0
            else:
                yield 1.0 - (2 * (cycle_position - midpoint) / descend_length)
            sample_n += 1



class Triangle(Sawtooth):
    def __init__(self, freq, **kwargs):
        kwargs['duty_cycle'] = 0.5
        super(Triangle, self).__init__(freq, **kwargs)


class WhiteNoise(SignalGenerator):
    def generate(self):
        while True:
            yield (random.random() * 2) - 1.0
