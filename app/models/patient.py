# app/models/patient.py
from app.extensions import db

class Patient(db.Model):
    __tablename__ = "patients"

    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    dni = db.Column(db.String(20))
    age = db.Column(db.Integer)
    gender = db.Column(db.Enum("M", "F", "O"))
    created_at = db.Column(db.DateTime, server_default=db.func.now())
