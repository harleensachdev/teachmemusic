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

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-default-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///your_database_name.db')
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
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('performances', lazy=True))
    score = db.relationship('Score', backref=db.backref('performances', lazy=True))

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

@app.route('/init_db')
def init_db():
    try:
        db.create_all()
        return 'Database initialized!'
    except Exception as e:
        return f'Error initializing database: {str(e)}'

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}", exc_info=True)
    db.session.rollback()
    return render_template('500.html'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
