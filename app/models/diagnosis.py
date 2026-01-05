from app.extensions import db

class Diagnosis(db.Model):
    __tablename__ = "diagnoses"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=False)

    image_path = db.Column(db.String(255), nullable=False)
    heatmap_path = db.Column(db.String(255), nullable=False)

    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Numeric(5, 2))
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    # Relationships
    patient = db.relationship("Patient", backref=db.backref("diagnoses", lazy=True))
    doctor = db.relationship("Doctor", backref=db.backref("diagnoses", lazy=True))