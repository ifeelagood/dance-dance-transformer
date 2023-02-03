import os
import json
import pathlib
import threading
import queue
import tqdm


import simfile 
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes
from pydub import AudioSegment


SIMFILE_EXTENSIONS = (".ssc", ".sm")
AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")
NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)


def locate_audio(song_path : pathlib.Path) -> pathlib.Path:
    """Locate audio file at the given path"""

    for file in os.listdir(song_path):
        for ext in AUDIO_EXTENSIONS:
            if file.endswith(ext):
                return song_path / file


def process_charts(config, simfile_obj, pack_name, wav_path, num_samples):
    """Process the given simfile object into a list of charts"""

    charts = []
    # iterate through charts
    for chart in simfile_obj.charts:
        # check if chart type is valid
        if chart.stepstype not in config.charts.types:
            continue
    
        # check if difficulty is valid
        if chart.difficulty not in config.charts.difficulties:
            continue
        
        # get note data and timing data
        note_data = NoteData(chart.notes)
        timing_data = TimingData(simfile_obj, chart)

        actions = [] # an integer representing (tap, hold, tail, roll)
        columns = [] # an integer representing the arrow direction
        beats = []   # the beat of the note
        timings = [] # the time in second of the note

        for timed_note in time_notes(note_data, timing_data):
            note = timed_note.note
            
            # skip invalid actions
            if note.note_type not in NOTES:
                continue                

            timings.append(timed_note.time)
            columns.append(note.column)
            actions.append(note.note_type.value)
            beats.append(float(note.beat))
            
        # get bpm timings
        bpms = [(int(x.beat), float(x.value)) for x in timing_data.bpms]
        
        # create a new chart dict

        new_chart = {
            "title": simfile_obj.title,
            "difficulty": chart.difficulty.lower(),
            "samplestart": float(simfile_obj.samplestart),
            "samplelength": float(simfile_obj.samplelength)s,
            "offset": float(simfile_obj.offset),
            "bpms": bpms,
            "pack": pack_name,
            "actions": actions,
            "columns": columns,
            "beats": beats,
            "timings": timings,
            "wav": wav_path.name,
            "samples": num_samples
        }
        
        # append to charts
        charts.append(new_chart)
        
    return charts


def process_song(config, pack_name, song_name):
    song_path = config.paths.raw / pack_name / song_name

    # Locate audio file
    audio_path = locate_audio(song_path)
    if audio_path is None:
        print(f"WARNING: {song_path} does not have an audio file")
        return

    # load audio file
    wav = AudioSegment.from_file(audio_path)

    # resample and convert to mono
    wav.set_frame_rate(config.audio.sample_rate)
    wav.set_channels(1)
    
    # get number of samples
    num_samples = len(wav.get_array_of_samples())

    # export wav to export directory
    wav_path = config.paths.wav / audio_path.with_suffix(".wav").name
    wav.export(wav_path, format="wav")

    # open the simfile
    simfile_obj, simfile_name = simfile.opendir(song_path)

    # process charts
    processed_charts = process_charts(config, simfile_obj, pack_name, wav_path, num_samples)
    return processed_charts


def process_packs(config, num_threads=16):
    """Process all packs in config.dataset.packs"""

    # add (pack_dir, song) to queue
    q = queue.Queue()
    for pack_name in config.dataset.packs:
        pack_path = config.paths.raw / pack_name
        for song_name in os.listdir(pack_path):
            q.put((pack_name, song_name))
    
    # store charts
    charts = []

    def worker(q):
        nonlocal charts
        while not q.empty():
            pack_name, song_name = q.get()
            charts.extend(process_song(config, pack_name, song_name))
            q.task_done()

    # start threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(q,))
        t.start()
        threads.append(t)
    
    # wait for threads to finish
    for t in threads:
        t.join()

    # write charts to json file
    with open(config.paths.charts, "w") as f:
        json.dump(charts, f, indent=4)

