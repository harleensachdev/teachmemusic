from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from templates.notedetection import detect_notes, detect_notes_from_musicxml, detect_notes_from_midi
from collections import defaultdict
from datetime import datetime
import librosa
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
from pydub import AudioSegment
import io
import logging
from flask_login import UserMixin, login_user, login_required, current_user, logout_user
from extensions import db, bcrypt, login_manager, migrate

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'xml', 'musicxml', 'mid', 'midi'}

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
migrate.init_app(app, db)

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model definitions
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

class PerformanceAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'), nullable=False)  # Change from note_data.id to score.id
    accuracy = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('performances', lazy=True))
    score = db.relationship('Score', backref=db.backref('performances', lazy=True))  # Update relationship

class Score(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    notes = db.relationship('NoteData', back_populates='score', cascade='all, delete-orphan')

class NoteData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    measure = db.Column(db.Integer, nullable=False)
    note_name = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.String(10), nullable=False)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'), nullable=False)
    score = db.relationship('Score', back_populates='notes')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return redirect(url_for('register'))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check email and password.', 'danger')

    return render_template('login.html')

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/upload_score", methods=['POST'])
@login_required
def upload_score():
    if 'music_score' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('dashboard'))

    file = request.files['music_score']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('dashboard'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            if filename.lower().endswith(('.xml', '.musicxml')):
                result = detect_notes_from_musicxml(file_path)
            elif filename.lower().endswith(('.mid', '.midi')):
                result = detect_notes_from_midi(file_path)
            else:
                result = detect_notes(file_path)
            score_id = store_notes(result['notes'])
            return redirect(url_for('display_score', image=filename, score_id=score_id))
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(url_for('dashboard'))
    else:
        flash('Invalid file type. Only PNG, JPG, JPEG, MusicXML, and MIDI files are allowed.', 'danger')
        return redirect(url_for('dashboard'))

def store_notes(notes):
    try:
        # Create a new Score object
        new_score = Score(title="Uploaded Score")  # You might want to get a title from the user
        db.session.add(new_score)
        db.session.flush()  # This will assign an ID to new_score without committing the transaction

        for note in notes:
            new_note = NoteData(
                measure=note['measure'],
                note_name=note['pitch'],
                duration=str(note['duration']),
                score_id=new_score.id  # Use the ID of the newly created Score
            )
            db.session.add(new_note)
        
        db.session.commit()
        logger.debug(f"Stored new score with id {new_score.id} and {len(notes)} notes")
        return new_score.id
    except Exception as e:
        logger.error(f'Error storing notes: {e}')
        db.session.rollback()
        raise

def quantize_duration(duration):
    """Convert note duration to VexFlow duration format."""
    # Duration mapping from numerical values to VexFlow format
    duration_map = {
        # Whole notes
        '4.0': 'w',     # Whole note
        '6.0': 'wd',    # Dotted whole
        '7.0': 'wdd',   # Double dotted whole
        
        # Half notes
        '2.0': 'h',     # Half note
        '3.0': 'hd',    # Dotted half
        '3.5': 'hdd',   # Double dotted half
        
        # Quarter notes
        '1.0': 'q',     # Quarter note
        '1.5': 'qd',    # Dotted quarter
        '1.75': 'qdd',  # Double dotted quarter
        
        # Eighth notes
        '0.5': '8',     # Eighth note
        '0.75': '8d',   # Dotted eighth
        '0.875': '8dd', # Double dotted eighth
        
        # Sixteenth notes
        '0.25': '16',   # Sixteenth note
        '0.375': '16d', # Dotted sixteenth
        '0.4375': '16dd', # Double dotted sixteenth
        
        # 32nd notes
        '0.125': '32',  # 32nd note
        '0.1875': '32d', # Dotted 32nd
        
        # Triplets
        '0.33333': '8tr',  # Eighth triplet
        '0.66667': 'qtr',  # Quarter triplet
        '1.33333': 'htr',  # Half triplet
        
        # Rests
        'rest_4.0': 'wr',   # Whole rest
        'rest_2.0': '2r',   # Half rest
        'rest_1.0': '4r',   # Quarter rest
        'rest_0.5': '8r',   # Eighth rest
        'rest_0.25': '16r', # Sixteenth rest
        'rest_0.125': '32r' # 32nd rest
    }
    
    # Convert duration to string for lookup
    duration_str = str(float(duration))
    
    # Handle rests
    if isinstance(duration, str) and duration.startswith('rest'):
        duration_str = f'rest_{duration.split("_")[1]}'
    
    # Return mapped duration or default to quarter note
    return duration_map.get(duration_str, 'q')

@app.route("/display_score")
@login_required
def display_score():
    score_id = request.args.get('score_id')
    notes = NoteData.query.filter_by(score_id=score_id).order_by(NoteData.measure, NoteData.id).all()
    image_filename = request.args.get('image', '')
    uploaded_image_url = url_for('static', filename=f'uploads/{image_filename}')
    
    vexflow_notes = defaultdict(list)
    for note in notes:
        vexflow_duration = quantize_duration(note.duration)
        
        if note.note_name.lower() == 'rest':
            note_key = "b/4"
            is_rest = True
        else:
            note_name = note.note_name[:-1].lower()
            octave = note.note_name[-1]
            note_key = f"{note_name}/{octave}"
            is_rest = False
        
        note_data = {
            "keys": [note_key],
            "duration": vexflow_duration,
            "is_rest": is_rest
        }
        vexflow_notes[note.measure].append(note_data)
    
    vexflow_notes_dict = {str(k): v for k, v in vexflow_notes.items()}
    
    print("VexFlow notes data:", vexflow_notes_dict)  # Debugging line
    
    return render_template('display_score.html', 
                           detected_notes=notes, 
                           uploaded_image_url=uploaded_image_url, 
                           vexflow_notes=vexflow_notes_dict,
                           time_signature="3/4",
                           score_id=score_id)

@app.route("/record_performance")
@login_required
def record_performance():
    # Get the latest score from the database
    latest_score = Score.query.order_by(Score.id.desc()).first()
    
    if not latest_score:
        flash('Please upload a score first', 'warning')
        return redirect(url_for('dashboard'))
    
    note_data = NoteData.query.filter_by(score_id=latest_score.id).order_by(NoteData.measure, NoteData.id).all()
    
    vexflow_notes = defaultdict(list)
    for note in note_data:
        vexflow_duration = quantize_duration(note.duration)
        
        if note.note_name.lower() == 'rest':
            note_key = "b/4"
            is_rest = True
        else:
            note_name = note.note_name[:-1].lower()
            octave = note.note_name[-1]
            note_key = f"{note_name}/{octave}"
            is_rest = False
        
        note_data = {
            "keys": [note_key],
            "duration": vexflow_duration,
            "is_rest": is_rest
        }
        vexflow_notes[note.measure].append(note_data)
    
    vexflow_notes_dict = {str(k): v for k, v in vexflow_notes.items()}
    
    print("VexFlow notes data:", vexflow_notes_dict)  # Debug print
    
    return render_template('record_performance.html', 
                         vexflow_notes=vexflow_notes_dict,
                         score_id=latest_score.id)

@app.route("/analyze_recording", methods=['POST'])
@login_required
def analyze_recording():
    logger.debug("Entering analyze_recording function")
    if 'audio' not in request.files:
        logger.error("No audio file provided in the request")
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    score_id = request.form.get('score_id')
    
    logger.debug(f"Received audio file and score_id: {score_id}")

    if not score_id:
        logger.error("No score ID provided in the request")
        return jsonify({'error': 'No score ID provided'}), 400

    # Check if the score exists
    score = Score.query.get(score_id)
    if not score:
        logger.error(f"No score found with id {score_id}")
        return jsonify({'error': f'No score found with id {score_id}'}), 404

    # Save the audio file temporarily
    temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_recording.wav')
    audio_file.save(temp_audio_path)
    logger.debug(f"Saved temporary audio file to {temp_audio_path}")

    try:
        # Convert the audio to WAV format
        audio = AudioSegment.from_file(temp_audio_path)
        audio.export(temp_audio_path, format="wav")
        logger.debug("Converted audio to WAV format")

        # Analyze the audio
        logger.debug("Starting audio analysis")
        accuracy, feedback, correct_notes, incorrect_notes = analyze_audio(temp_audio_path, score_id)
        logger.debug(f"Audio analysis complete. Accuracy: {accuracy}, Correct notes: {len(correct_notes)}, Incorrect notes: {len(incorrect_notes)}")

        # Store the analysis results
        try:
            analysis = PerformanceAnalysis(
                user_id=current_user.id,
                score_id=int(score_id),
                accuracy=accuracy,
                feedback=feedback
            )
            db.session.add(analysis)
            db.session.commit()
            logger.debug("Analysis results stored in database")
        except Exception as db_error:
            logger.error(f"Database error: {str(db_error)}")
            db.session.rollback()

        response_data = {
            'feedback': feedback,
            'accuracy': accuracy,
            'correct_notes': correct_notes,
            'incorrect_notes': incorrect_notes
        }
        logger.debug(f"Sending response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in analyze_recording: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        # Remove the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            logger.debug(f"Removed temporary audio file: {temp_audio_path}")

def analyze_audio(audio_path, score_id):
    logger.debug(f"Entering analyze_audio function with audio_path: {audio_path} and score_id: {score_id}")
    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_wav(audio_path)
        samples = audio.get_array_of_samples()
        logger.debug(f"Loaded audio file. Duration: {len(audio)/1000}s, Sample rate: {audio.frame_rate}Hz")
        
        # Convert to numpy array
        y = np.array(samples).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        sr = audio.frame_rate

        # Extract pitch and onset information
        logger.debug("Extracting pitch and onset information")
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        logger.debug(f"Detected {len(onsets)} onsets")

        # Detect notes
        detected_notes = []
        for onset in onsets:
            start_frame = librosa.time_to_frames(onset, sr=sr)
            end_frame = start_frame + sr // 2  # Analyze 0.5 seconds after onset
            pitch_segment = pitches[:, start_frame:end_frame]
            mag_segment = magnitudes[:, start_frame:end_frame]

            pitch_index = np.unravel_index(mag_segment.argmax(), mag_segment.shape)
            freq = pitch_segment[pitch_index]
            note = librosa.hz_to_note(freq)
            detected_notes.append((note, onset))
        logger.debug(f"Detected {len(detected_notes)} notes: {detected_notes}")

        # Get the original score
        original_score = NoteData.query.filter_by(score_id=score_id).order_by(NoteData.measure, NoteData.id).all()
        logger.debug(f"Retrieved original score with {len(original_score)} notes")

        if not original_score:
            logger.error(f"No score found with id {score_id}")
            raise ValueError(f"No score found with id {score_id}")

        # Compare detected notes with the original score
        correct_notes = []
        incorrect_notes = []
        original_index = 0
        detected_index = 0

        while original_index < len(original_score) and detected_index < len(detected_notes):
            original_note = original_score[original_index]
            detected_note, detected_time = detected_notes[detected_index]

            logger.debug(f"Comparing original note {original_note.note_name} with detected note {detected_note}")
            if original_note.note_name.lower() == detected_note.lower():
                correct_notes.append(original_index)
                original_index += 1
                detected_index += 1
            else:
                incorrect_notes.append(original_index)
                original_index += 1

        # Calculate accuracy
        accuracy = len(correct_notes) / len(original_score) if original_score else 0
        logger.debug(f"Calculated accuracy: {accuracy}")

        # Generate feedback
        feedback = generate_feedback(accuracy, correct_notes, incorrect_notes, original_score)
        logger.debug(f"Generated feedback: {feedback}")

        logger.debug(f"Detected notes: {detected_notes}")
        logger.debug(f"Original score: {[(note.note_name, note.measure) for note in original_score]}")
        
        logger.debug(f"Correct notes: {correct_notes}")
        logger.debug(f"Incorrect notes: {incorrect_notes}")
        logger.debug(f"Accuracy: {accuracy}")

        return accuracy, feedback, correct_notes, incorrect_notes
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}", exc_info=True)
        raise

def generate_feedback(accuracy, correct_notes, incorrect_notes, original_score):
    feedback = f"Your overall accuracy was {accuracy:.2%}. "

    if accuracy >= 0.9:
        feedback += "Excellent job! "
    elif accuracy >= 0.7:
        feedback += "Good work! "
    elif accuracy >= 0.5:
        feedback += "Nice effort! "
    else:
        feedback += "Keep practicing! "

    if incorrect_notes:
        feedback += "You might want to focus on the following measures: "
        incorrect_measures = set(original_score[i].measure for i in incorrect_notes)

    return feedback

@app.route("/rhythm_check")
@login_required
def rhythm_check():
    try:
        # Get the latest score from the database
        latest_score = Score.query.order_by(Score.id.desc()).first()
        
        if not latest_score:
            # Provide default notes if no score exists
            vexflow_notes = {
                '1': [
                    {"keys": ["c/4"], "duration": "q"},
                    {"keys": ["d/4"], "duration": "q"},
                    {"keys": ["e/4"], "duration": "q"},
                    {"keys": ["f/4"], "duration": "q"}
                ]
            }
        else:
            # Get notes from database
            notes = NoteData.query.filter_by(score_id=latest_score.id).order_by(NoteData.measure, NoteData.id).all()
            
            # Convert to VexFlow format
            vexflow_notes = defaultdict(list)
            for note in notes:
                vexflow_duration = quantize_duration(note.duration)
                
                if note.note_name.lower() == 'rest':
                    note_key = "b/4"
                    is_rest = True
                else:
                    note_name = note.note_name[:-1].lower()
                    octave = note.note_name[-1]
                    note_key = f"{note_name}/{octave}"
                    is_rest = False
                
                note_data = {
                    "keys": [note_key],
                    "duration": vexflow_duration,
                    "is_rest": is_rest
                }
                vexflow_notes[str(note.measure)].append(note_data)
        
        # Debug print
        print("VexFlow notes being sent to template:", vexflow_notes)
        
        return render_template('rhythm_check.html', vexflow_notes=vexflow_notes)

    except Exception as e:
        logger.error(f"Error in rhythm_check route: {str(e)}", exc_info=True)
        flash('Error loading rhythm check. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

@app.route("/save_rhythm_score", methods=['POST'])
@login_required
def save_rhythm_score():
    try:
        data = request.get_json()
        accuracy = data.get('accuracy')
        
        if accuracy is not None:
            # Store the rhythm analysis results
            analysis = PerformanceAnalysis(
                user_id=current_user.id,
                score_id=data.get('score_id', 1),  # Use a default score_id if none provided
                accuracy=float(accuracy),
                feedback=f"Rhythm accuracy: {accuracy}%"
            )
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify({
                'status': 'success',
                'message': 'Rhythm score saved successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No accuracy score provided'
            }), 400

    except Exception as e:
        logger.error(f"Error saving rhythm score: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': 'Error saving rhythm score'
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # This will create the tables based on your models
    app.run(debug=True)
