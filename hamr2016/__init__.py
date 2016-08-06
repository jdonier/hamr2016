from __future__ import print_function, division
import midi
import matplotlib.pyplot as plt
import numpy as np
import librosa
from IPython import display
from warnings import warn


class Track:
    def __init__(self, notes, resolution, tempo):
        self.notes = notes
        self.resolution = resolution
        self.tempo = tempo
    
    @staticmethod
    def from_file(filename, track_id=1, tempo=120):
        pattern = midi.read_midifile(filename)
        pattern.make_ticks_abs()
        # Process the metadata
        if pattern.format:
            for event in pattern[0]:
                if isinstance(event, midi.SetTempoEvent):
                    tempo = event.get_bpm()
        
        # Process the track
        notes = []
        current = {}
        
        for event in pattern[track_id]:
            if isinstance(event, midi.NoteOnEvent):
                current[event.get_pitch()] = event.tick
            elif isinstance(event, midi.NoteOffEvent):
                pitch = event.get_pitch()
                assert pitch in current, "cannot stop note that isn't playing"
                
                # Add the note
                notes.append((pitch, current[pitch], event.tick))
                
                # Delete the note from the currently playing notes
                del current[pitch]
                    
        notes = np.rec.array(notes, names=['pitch', 'start', 'end'])
        return Track(notes, pattern.resolution, tempo)

    @property
    def duration(self):
        return self.tick_to_time(np.max(self.notes.end))
    
    def tick_to_beat(self, tick):
        return tick / self.resolution
    
    def beat_to_tick(self, beat):
        return np.round(beat * self.resolution).astype(int)
    
    def tick_to_time(self, tick):
        return self.tick_to_beat(tick) * 60 / self.tempo
    
    def time_to_tick(self, time):
        return self.beat_to_tick(self.tempo * time / 60)
    
    def synthesize(self, sample_rate=44100):
        samples = np.zeros(np.round(np.ceil(sample_rate * self.duration)).astype(int))
        
        for pitch, start, end in self.notes:
            i = np.round(self.tick_to_time(start) * sample_rate).astype(int)
            j = np.round(self.tick_to_time(end) * sample_rate).astype(int)
            buffer = np.sin(librosa.midi_to_hz(pitch) * 2 * np.pi * np.arange(j - i) / sample_rate)
            buffer *= 1 - np.linspace(0, 1, len(buffer)) ** 2
            samples[i:j] += buffer
        
        return display.Audio(samples, rate=sample_rate)
    
    def to_matrix(self, lower_bound=21, upper_bound=109, resolution=4, strict=True):
        # Get the total number of beats
        num_beats = np.ceil(self.tick_to_beat(np.max(self.notes.end)) * resolution).astype(int)
        num_midis = upper_bound - lower_bound
        matrix = np.zeros((num_midis, num_beats))
        # Iterate over all notes
        for pitch, start, end in self.notes:
            if pitch < lower_bound:
                message = "pitch of {} is smaller than the lower bound {}".format(pitch, lower_bound)
                if strict:
                    raise ValueError(message)
                else:
                    warn(message)
            # Compute the offsets
            start = np.round(self.tick_to_beat(start) * resolution).astype(int)
            end = np.round(self.tick_to_beat(end) * resolution).astype(int)
            # Set the matrix entries
            matrix[pitch - lower_bound, start:end] = 1
        
        return Matrix(matrix, lower_bound, resolution, self.tempo)


class Matrix:
    def __init__(self, values, lower_bound=21, resolution=8, tempo=120):
        self.values = values
        self.lower_bound = lower_bound
        self.resolution = resolution
    
    def show(self, ax=None, **kwargs):
        kwargs_default = {
            'cmap': 'binary',
            'origin': 'lower',
            'interpolation': 'nearest',
        }
        kwargs_default.update(kwargs)
        ax = ax or plt.gca()
        return ax.imshow(self.values, **kwargs_default)
    
    def pixel_to_beat(self, pixel):
        return pixel * self.resolution
    
    def beat_to_pixel(self, beat):
        return np.round(beat / self.resolution)
    
    def to_track(self, resolution=960):
        for column in self.values.T:
            notes = self.lower_bound + np.nonzero(column)[0]


if __name__ == '__main__':
    filename = 'for_elise_by_beethoven.mid'
    track = Track.from_file(filename)
    matrix = track.to_matrix()
    matrix.show()
    plt.show()