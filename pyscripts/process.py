#!/usr/bin/python

import os
import json
import argparse


import simfile 
from simfile.notes import NoteData, NoteType
from simfile.timing import TimingData
from simfile.notes.timed import time_notes



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

def process_simfile(simfile_obj):
    """Process the given simfile object into a list of charts"""

    charts = []
    # iterate through charts
    for chart in simfile_obj.charts:
        # check if chart tpye is valid
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
        phase = []
        timings = []

        for note in time_notes(note_data, timing_data):
            # skip notes that are not taps, holds, or rolls.
            if note.note_type not in NOTES:
                continue

            columns.append(note.column)
            actions.append(note.note_type.value)
            timings.append(note.time)
            beats.append(note.beat)
            
    


def process_song(root, path):
    """Process the song at the given path"""

    # locate the audio file
    audio_path = locate_audio(root, path)
    if audio_path is None:
        print(f"WARNING: {path} does not have an audio file")
        return


    # open the simfile
    simfile_obj, simfile_name = simfile.opendir(os.path.join(root, path))



    # process the simfile
    process_simfile(simfile_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stepmania songs")
    parser.add_argument('path', metavar='PATH', type=str, help="Path to directory containing songs")
    args = parser.parse_args()

    # process all songs in the given directory
    for root, dirs, files in os.walk(args.path):
        for d in dirs:
            process_song(root, d)