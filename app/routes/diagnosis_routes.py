import os
import uuid
import random
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.extensions import db
from app.models.diagnosis import Diagnosis
from app.models.patient import Patient

# Intento de importación condicional para evitar fallos si TF no está instalado
try:
    from app.ml.covid_predictor import predict_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Advertencia: covid_predictor no disponible (TensorFlow faltante).")

diagnosis_bp = Blueprint("diagnosis", __name__)

UPLOAD_FOLDER = "media/uploads/x-ray"
HEATMAP_FOLDER = "media/heatmap"

@diagnosis_bp.route("/", methods=["GET"])
@jwt_required()
def get_diagnoses():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Filtros
    patient_id = request.args.get('patient_id')
    result_filter = request.args.get('result')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    query = Diagnosis.query

    if patient_id:
        query = query.filter_by(patient_id=patient_id)
    
    if result_filter:
        if request.args.get('result') == "NO COVID":
            query = query.filter(Diagnosis.result.ilike("%NORMAL%"))
        else:
            query = query.filter(Diagnosis.result.ilike(f"%{result_filter}%"))
        
    if date_from:
        query = query.filter(Diagnosis.created_at >= date_from)
        
    if date_to:
        query = query.filter(Diagnosis.created_at <= date_to)

    # Ordenar por defecto descendente
    query = query.order_by(Diagnosis.id.desc())

    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    diagnoses = pagination.items
    
    base_url = request.host_url.rstrip("/")
    data = []
    
    for d in diagnoses:
        # Obtener nombre del paciente si es posible
        patient_name = d.patient.full_name if d.patient else "Desconocido"
        
        data.append({
            "id": d.id,
            "patient_name": patient_name,
            "result": d.result,
            "confidence": float(d.confidence),
            "image_url": f"{base_url}/media/{d.image_path}",
            "heatmap_url": f"{base_url}/media/{d.heatmap_path}",
            "created_at": d.created_at.isoformat() if d.created_at else None
        })
        
    return jsonify({
        "page": pagination.page,
        "per_page": per_page,
        "total": pagination.total,
        "data": data
    }), 200

@diagnosis_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    # 1️⃣ Validar datos básicos
    if "image" not in request.files:
        return {"error": "No image provided"}, 400
    
    patient_id = request.form.get("patient_id")
    disease_type = request.form.get("disease_type", "COVID") # Default a COVID

    if not patient_id:
        return {"error": "patient_id required"}, 400
        
    doctor_id = get_jwt_identity()

    # Verificar que el paciente exista
    patient = Patient.query.get(patient_id)
    if not patient:
        return {"error": "Paciente no encontrado"}, 404

    image = request.files["image"]

    # Preparar carpetas
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    heatmap_path = os.path.join(HEATMAP_FOLDER, filename)
    
    # Guardar imagen original
    image.save(image_path)

    #  Lógica de Predicción según enfermedad
    label = "Desconocido"
    confidence = 0.0
    
    # Rutas relativas para DB
    db_image_path = f"uploads/x-ray/{filename}"
    db_heatmap_path = f"heatmap/{filename}"

    try:
        if disease_type.upper() == "COVID":
            # Predicción REAL
            if TF_AVAILABLE:
                label, confidence = predict_image(image_path, heatmap_path)
            else:
                label = "MOCK_RESULT"
                confidence = 90.0
                import shutil
                shutil.copy(image_path, heatmap_path)
        
        elif disease_type.upper() in ["IRA", "EDA", "HIPERTENSION", "DIABETES"]:
            # --- MOCK / SIMULACIÓN ---
            import shutil
            shutil.copy(image_path, heatmap_path)
            
            possible_outcomes = ["Positivo", "Negativo", "Riesgo Alto", "Riesgo Bajo"]
            label = random.choice(possible_outcomes)
            confidence = round(random.uniform(70.0, 99.0), 2)
            
        else:
            return {"error": f"Tipo de enfermedad '{disease_type}' no soportado"}, 400

        # Guardamos el resultado. Si es COVID, guardamos solo "COVID" o "NORMAL" según el label.
        # Si es otro, quizás queramos guardar el tipo. Pero para cumplir la spec, el output es 'result'.
        # Guardaremos el label directo.
        final_result = label

        #  Guardar en BD
        diagnosis = Diagnosis(
            patient_id=patient_id,
            doctor_id=doctor_id,
            image_path=db_image_path,
            heatmap_path=db_heatmap_path,
            result=final_result,
            confidence=confidence
        )

        db.session.add(diagnosis)
        db.session.commit()

        base_url = request.host_url.rstrip("/")

        return jsonify({
            "result": final_result,
            "confidence": float(confidence),
            "image_url": f"{base_url}/media/{db_image_path}",
            "heatmap_url": f"{base_url}/media/{db_heatmap_path}"
        })

    except Exception as e:
        print(f"Error en predicción: {e}")
        return {"error": "Error interno durante el procesamiento"}, 500

@diagnosis_bp.route("/patient/<int:patient_id>", methods=["GET"])
@jwt_required()
def get_patient_history(patient_id):
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    pagination = Diagnosis.query.filter_by(patient_id=patient_id).order_by(Diagnosis.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    diagnoses = pagination.items
    
    base_url = request.host_url.rstrip("/")
    result = []
    
    for d in diagnoses:
        result.append({
            "id": d.id,
            "date": "Fecha no registrada", 
            "result": d.result,
            "confidence": float(d.confidence),
            "image_url": f"{base_url}/media/{d.image_path}",
            "heatmap_url": f"{base_url}/media/{d.heatmap_path}"
        })
        
    return jsonify({
        "items": result,
        "total": pagination.total,
        "pages": pagination.pages,
        "current_page": pagination.page
    }), 200