from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.models.patient import Patient
from app.models.diagnosis import Diagnosis
from app.extensions import db
from sqlalchemy import func
from datetime import datetime, timedelta

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/summary', methods=['GET'])
@jwt_required()
def get_dashboard_summary():
    doctor_id = get_jwt_identity()
    

    total_patients = Patient.query.filter_by(doctor_id=doctor_id).count()
    male_patients = Patient.query.filter_by(doctor_id=doctor_id, gender='M').count()
    female_patients = Patient.query.filter_by(doctor_id=doctor_id, gender='F').count()
    other_patients = Patient.query.filter_by(doctor_id=doctor_id, gender='O').count()
    
    avg_age_query = db.session.query(func.avg(Patient.age)).filter_by(doctor_id=doctor_id).scalar()
    average_age = int(avg_age_query) if avg_age_query else 0


    total_diagnoses = Diagnosis.query.filter_by(doctor_id=doctor_id).count()
    covid_positive = Diagnosis.query.filter_by(doctor_id=doctor_id, result='COVID').count()
    covid_negative = Diagnosis.query.filter_by(doctor_id=doctor_id, result='NORMAL').count()
    
    if total_diagnoses > 0:
        positive_rate = round((covid_positive / total_diagnoses) * 100, 1)
    else:
        positive_rate = 0


    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_diagnoses_count = Diagnosis.query.filter_by(doctor_id=doctor_id)\
        .filter(Diagnosis.created_at >= seven_days_ago).count()
        
    return jsonify({
        "patients": {
            "total": total_patients,
            "male": male_patients,
            "female": female_patients,
            "other": other_patients,
            "average_age": average_age
        },
        "diagnoses": {
            "total": total_diagnoses,
            "covid_positive": covid_positive,
            "covid_negative": covid_negative,
            "positive_rate": positive_rate
        },
        "recent_activity": {
            "diagnoses_last_7_days": recent_diagnoses_count
        }
    }), 200
