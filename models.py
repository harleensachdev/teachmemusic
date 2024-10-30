from flask_login import UserMixin
from extensions import db
from datetime import datetime

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
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    notes = db.relationship('NoteData', back_populates='score', cascade='all, delete-orphan')
    user = db.relationship('User', backref=db.backref('scores', lazy=True))

class NoteData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    measure = db.Column(db.Integer, nullable=False)
    note_name = db.Column(db.String(10), nullable=False)
    duration = db.Column(db.String(10), nullable=False)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'), nullable=False)
    score = db.relationship('Score', back_populates='notes')