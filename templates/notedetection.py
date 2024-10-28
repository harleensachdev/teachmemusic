import cv2
import numpy as np
from music21 import converter, note, stream, environment, metadata, instrument, key

# Set up Music21 environment to not use external software
us = environment.UserSettings()
us['musicxmlPath'] = None
us['musescoreDirectPNGPath'] = None

def detect_notes(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to find note heads
    note_heads = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.8 <= w/h <= 1.2 and 10 <= w <= 30:  # Adjust these values based on your images
            note_heads.append((x, y))
    
    # Sort note heads by x-coordinate
    note_heads.sort(key=lambda p: p[0])
    
    # Create a music21 stream
    s = stream.Stream()
    
    # Add notes to the stream
    for i, (x, y) in enumerate(note_heads):
        # This is a simplified pitch detection. You might need to adjust this based on your staff lines.
        pitch = 60 + (img.shape[0] - y) // 10  # Middle C (60) as reference
        n = note.Note(pitch)
        n.quarterLength = 1  # Assuming all notes are quarter notes
        s.append(n)
    
    # Add metadata
    s.metadata = metadata.Metadata()
    s.metadata.title = "Detected Score"
    
    # Extract relevant information
    notes = []
    for i, elem in enumerate(s.recurse().notesAndRests):
        if isinstance(elem, note.Note):
            notes.append({
                'pitch': elem.nameWithOctave,
                'duration': elem.quarterLength,
                'measure': i // 4 + 1  # Assuming 4/4 time signature
            })
    
    time_signature = "4/4"  # Assuming 4/4 time signature
    key_signature = s.analyze('key').tonic.name + " " + s.analyze('key').mode
    clef = "G"  # Assuming treble clef
    
    return {
        'notes': notes,
        'time_signature': time_signature,
        'key_signature': key_signature,
        'clef': clef
    }

def detect_notes_from_musicxml(file_path):
    # Parse the MusicXML file
    score = converter.parse(file_path)
    
    # Extract relevant information
    notes = []
    for i, elem in enumerate(score.recurse().notesAndRests):
        if isinstance(elem, note.Note):
            notes.append({
                'pitch': elem.nameWithOctave,
                'duration': elem.quarterLength,
                'measure': elem.measureNumber
            })
        elif isinstance(elem, note.Rest):
            notes.append({
                'pitch': 'Rest',
                'duration': elem.quarterLength,
                'measure': elem.measureNumber
            })
    
    # Get time signature
    time_signature = score.getTimeSignatures()[0].ratioString if score.getTimeSignatures() else "4/4"
    
    # Get key signature
    key_signature = score.analyze('key').tonic.name + " " + score.analyze('key').mode
    
    # Get clef
    clef = score.parts[0].measure(1).clef.sign if score.parts[0].measure(1).clef else "G"
    
    return {
        'notes': notes,
        'time_signature': time_signature,
        'key_signature': key_signature,
        'clef': clef
    }

def detect_notes_from_midi(file_path):
    # Parse the MIDI file
    midi = converter.parse(file_path)
    
    # Extract relevant information
    notes = []
    for part in midi.parts:
        for i, elem in enumerate(part.recurse().notesAndRests):
            if isinstance(elem, note.Note):
                notes.append({
                    'pitch': elem.nameWithOctave,
                    'duration': elem.quarterLength,
                    'measure': elem.measureNumber if elem.measureNumber else i // 4 + 1  # Fallback if measure number is not available
                })
            elif isinstance(elem, note.Rest):
                notes.append({
                    'pitch': 'Rest',
                    'duration': elem.quarterLength,
                    'measure': elem.measureNumber if elem.measureNumber else i // 4 + 1
                })
    
    # Get time signature (if available, otherwise assume 4/4)
    time_signatures = midi.getTimeSignatures()
    time_signature = time_signatures[0].ratioString if time_signatures else "4/4"
    
    # Get key signature (if available, otherwise analyze)
    key_signature = None
    for element in midi.flatten():
        if isinstance(element, key.KeySignature):
            key_signature = element
            break
    
    if key_signature:
        key_str = key_signature.asKey().tonic.name + " " + key_signature.asKey().mode
    else:
        key_str = midi.analyze('key').tonic.name + " " + midi.analyze('key').mode
    
    # Get clef (MIDI doesn't specify clef, so we'll assume treble)
    clef = "G"
    
    return {
        'notes': notes,
        'time_signature': time_signature,
        'key_signature': key_str,
        'clef': clef
    }
