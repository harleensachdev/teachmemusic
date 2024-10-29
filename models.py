from flask_sqlalchemy import SQLAlchemy
from app import db
from flask_login import UserMixin
from models import User  # Import all your models
import os
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

db = SQLAlchemy()

# Add this near the top of your configuration
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///your_database_name.db'

class NoteData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    score_id = db.Column(db.Integer, db.ForeignKey('score.id'))  # This line might be missing
    measure = db.Column(db.Integer)
    note_name = db.Column(db.String)
    duration = db.Column(db.Float)
    # Remove the score_id field from here

    # ... any other fields or methods ...

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'

def init_db():
    with app.app_context():
        try:
            db.create_all()
            logging.info("Database tables created successfully")
            return 'Database initialized!'
        except Exception as e:
            logging.error(f"Error creating database tables: {str(e)}")
            return f'Error initializing database: {str(e)}'
