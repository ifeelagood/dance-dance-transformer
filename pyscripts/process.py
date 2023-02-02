import os
import json
import tqdm

import simfile 
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes
from pydub import AudioSegment

SAMPLE_RATE = 44100

SIMFILE_EXTENSIONS = (".ssc", ".sm")
AUDIO_EXTENSIONS = (".wav", ".ogg", ".mp3")
DIFFICULTIES = ("beginner", "easy", "medium", "hard", "challenge")
CHART_TYPES = ("dance-single")
NOTES = (NoteType.TAP, NoteType.HOLD_HEAD, NoteType.TAIL, NoteType.ROLL_HEAD)

def is_simfile(path):
    """Returns true if the path is a simfile"""
    return path.endswith(".sm") or path.endswith(".ssc")

def is_audio(path):
    """Returns true if the path is an audio file"""
    return path.endswith(".mp3") or path.endswith(".ogg") or path.endswith(".wav")

def locate_audio(root, path):
    """Attempt to locate audio file at the given path"""

    # locate the audio file
    audio_path = None
    for ext in AUDIO_EXTENSIONS:
        audio_path = os.path.join(root, path, path + ext) 
        if os.path.exists(audio_path):
            break

    return audio_path

def process_simfile(simfile_obj, pack, wav_path):
    """Process the given simfile object into a list of charts"""

    charts = []
    # iterate through charts
    for chart in simfile_obj.charts:
        # check if chart type is valid
        if chart.stepstype not in CHART_TYPES:
            continue
    
        # check if difficulty is valid
        if chart.difficulty.lower() not in DIFFICULTIES:
            continue
        
        # get note data and timing data
        note_data = NoteData(chart.notes)
        timing_data = TimingData(simfile_obj, chart)

        actions = []
        columns = []
        beats = []
        timings = []

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
            "samplestart": simfile_obj.samplestart,
            "samplelength": simfile_obj.samplelength,
            "offset": simfile_obj.offset,
            "bpms": bpms,
            "pack": pack,
            "wav": wav_path,
            "actions": actions,
            "columns": columns,
            "beats": beats,
            "timings": timings,
        }
        
        # append to charts
        charts.append(new_chart)
        
    return charts

def process_packs(packs, data_dir="data"):
    # create output dirs
    wav_path = os.path.join(data_dir, "wavs")
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
    
    # store charts
    charts = []
    
    # process all songs in the given directory
    
    for pack in packs:
        for root, dirs, files in os.walk(os.path.join(data_dir, "raw", pack)):
            for d in tqdm.tqdm(list(dirs), desc=pack):
                    # locate the audio file
                    audio_path = locate_audio(root, d)
                    if audio_path is None:
                        print(f"WARNING: {d} does not have an audio file")
                        return

                    # resample audio file and export it
                    wav = AudioSegment.from_file(audio_path)
                    wav.set_frame_rate(SAMPLE_RATE) # resample
                    wav.set_channels(1) # to monophonic
                    
                    export_path = os.path.join(data_dir, "wavs", os.path.basename(audio_path)[:-4] + ".wav")
                    wav.export(export_path, format="wav")


                    # open the simfile
                    simfile_obj, simfile_name = simfile.opendir(os.path.join(root, d))

                    # process the simfile
                    charts.extend(process_simfile(simfile_obj, pack, export_path))


    # write charts.
    with open(os.path.join(data_dir, "charts.json"), 'w') as f:
        json.dump(charts, f)

if __name__ == "__main__":
    process_packs(["fraxtil", "itg"])