from flask_sqlalchemy import SQLAlchemy
from app import db

db = SQLAlchemy()

class NoteData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'))  # This line might be missing
    measure = db.Column(db.Integer)
    note_name = db.Column(db.String)
    duration = db.Column(db.Float)
    # Remove the score_id field from here

    # ... any other fields or methods ...
