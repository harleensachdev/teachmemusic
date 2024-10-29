from flask import Flask, render_template, redirect, url_for, request, flash, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
import os
from models import db
from werkzeug.utils import secure_filename
from templates.notedetection import detect_notes, detect_notes_from_musicxml, detect_notes_from_midi
from collections import defaultdict
from datetime import datetime
import librosa
import numpy as np
from scipy.signal import find_peaks
import soundfile as sf
from pydub import AudioSegment
from flask_bcrypt import Bcrypt
import io
import logging
from flask_login import UserMixin, login_user, login_required, current_user, logout_user, LoginManager
from extensions import db, bcrypt, login_manager, migrate
from models import User

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Database Configuration
    database_url = os.environ.get('DATABASE_URL')
    if database_url:
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    else:
        database_url = 'sqlite:///site.db'

    # App Configuration
    app.config.update(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'your-default-secret-key'),
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER='uploads/',
        ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'xml', 'musicxml', 'mid', 'midi'},
        MAX_CONTENT_LENGTH=16 * 1024 * 1024
    )

    # Initialize extensions
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    login_manager.login_view = 'login'
    login_manager.login_message_category = 'info'

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Create all tables
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {str(e)}")

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

    # Helper Functions
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

    def store_notes(notes):
        try:
            new_score = Score(title="Uploaded Score")
            db.session.add(new_score)
            db.session.flush()

            for note in notes:
                new_note = NoteData(
                    measure=note['measure'],
                    note_name=note['pitch'],
                    duration=str(note['duration']),
                    score_id=new_score.id
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
        duration_map = {
            '4.0': 'w', '6.0': 'wd', '7.0': 'wdd',
            '2.0': 'h', '3.0': 'hd', '3.5': 'hdd',
            '1.0': 'q', '1.5': 'qd', '1.75': 'qdd',
            '0.5': '8', '0.75': '8d', '0.875': '8dd',
            '0.25': '16', '0.375': '16d', '0.4375': '16dd',
            '0.125': '32', '0.1875': '32d',
            '0.33333': '8tr', '0.66667': 'qtr', '1.33333': 'htr',
            'rest_4.0': 'wr', 'rest_2.0': '2r', 'rest_1.0': '4r',
            'rest_0.5': '8r', 'rest_0.25': '16r', 'rest_0.125': '32r'
        }
        duration_str = str(float(duration))
        if isinstance(duration, str) and duration.startswith('rest'):
            duration_str = f'rest_{duration.split("_")[1]}'
        return duration_map.get(duration_str, 'q')

    # Routes
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    @app.route("/")
    def home():
        return render_template('home.html')

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
        
        return render_template('display_score.html', 
                             detected_notes=notes, 
                             uploaded_image_url=uploaded_image_url, 
                             vexflow_notes=vexflow_notes_dict,
                             time_signature="3/4",
                             score_id=score_id)

    @app.route("/record_performance")
    @login_required
    def record_performance():
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
        
        if not score_id:
            logger.error("No score ID provided in the request")
            return jsonify({'error': 'No score ID provided'}), 400

        score = Score.query.get(score_id)
        if not score:
            logger.error(f"No score found with id {score_id}")
            return jsonify({'error': f'No score found with id {score_id}'}), 404

        temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_recording.wav')
        audio_file.save(temp_audio_path)

        try:
            audio = AudioSegment.from_file(temp_audio_path)
            audio.export(temp_audio_path, format="wav")
            
            accuracy, feedback, correct_notes, incorrect_notes = analyze_audio(temp_audio_path, score_id)
            
            analysis = PerformanceAnalysis(
                user_id=current_user.id,
                score_id=int(score_id),
                accuracy=accuracy,
                feedback=feedback
            )
            db.session.add(analysis)
            db.session.commit()

            response_data = {
                'feedback': feedback,
                'accuracy': accuracy,
                'correct_notes': correct_notes,
                'incorrect_notes': incorrect_notes
            }
            return jsonify(response_data)
        except Exception as e:
            logger.error(f"Error in analyze_recording: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    @app.route("/rhythm_check")
    @login_required
    def rhythm_check():
        try:
            latest_score = Score.query.order_by(Score.id.desc()).first()
            
            if not latest_score:
                vexflow_notes = {
                    '1': [
                        {"keys": ["c/4"], "duration": "q"},
                        {"keys": ["d/4"], "duration": "q"},
                        {"keys": ["e/4"], "duration": "q"},
                        {"keys": ["f/4"], "duration": "q"}
                    ]
                }
            else:
                notes = NoteData.query.filter_by(score_id=latest_score.id).order_by(NoteData.measure, NoteData.id).all()
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
                analysis = PerformanceAnalysis(
                    user_id=current_user.id,
                    score_id=data.get('score_id', 1),
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

    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))