from flask import Blueprint, request, jsonify
from app.models.doctor import Doctor
from app.extensions import db
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    # Validar campos requeridos
    required_fields = ['email', 'password', 'full_name']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Falta el campo {field}"}), 400

    if Doctor.query.filter_by(email=data['email']).first():
        return jsonify({"error": "El email ya está registrado"}), 400

    hashed_password = generate_password_hash(data['password'])
    
    new_doctor = Doctor(
        email=data['email'],
        password=hashed_password,
        full_name=data['full_name'],
        specialty=data.get('specialty', '')
    )

    try:
        db.session.add(new_doctor)
        db.session.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({"error": "Email y contraseña son requeridos"}), 400

    doctor = Doctor.query.filter_by(email=data['email']).first()

    if doctor and check_password_hash(doctor.password, data['password']):
        access_token = create_access_token(identity=str(doctor.id))
        return jsonify({
            "access_token": access_token
        }), 200

    return jsonify({"error": "Credenciales inválidas"}), 401

@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def profile():
    current_doctor_id = get_jwt_identity()
    doctor = Doctor.query.get(current_doctor_id)

    if not doctor:
        return jsonify({"error": "Doctor no encontrado"}), 404

    return jsonify({
        "id": doctor.id,
        "full_name": doctor.full_name,
        "email": doctor.email,
        "specialty": doctor.specialty,
        "is_active": doctor.is_active
    }), 200

@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    current_doctor_id = get_jwt_identity()
    doctor = Doctor.query.get(current_doctor_id)

    if not doctor:
        return jsonify({"error": "Doctor no encontrado"}), 404
    
    if not doctor.is_active:
        return jsonify({"error": "La cuenta del doctor está desactivada"}), 403

    data = request.get_json()
    
    # Actualizar nombre completo
    if 'full_name' in data:
        doctor.full_name = data['full_name']
    
    # Actualizar email (verificar que no esté en uso)
    if 'email' in data and data['email'] != doctor.email:
        existing_doctor = Doctor.query.filter_by(email=data['email']).first()
        if existing_doctor:
            return jsonify({"detail": "El email ya está en uso"}), 400
        doctor.email = data['email']
    
    # Cambiar contraseña si se proporciona
    if 'new_password' in data and data['new_password']:
        # Verificar que se proporcione la contraseña actual
        if 'current_password' not in data or not data['current_password']:
            return jsonify({"detail": "Debe proporcionar la contraseña actual"}), 400
        
        # Verificar que la contraseña actual sea correcta
        if not check_password_hash(doctor.password, data['current_password']):
            return jsonify({"detail": "La contraseña actual es incorrecta"}), 400
        
        # Actualizar con la nueva contraseña
        doctor.password = generate_password_hash(data['new_password'])
        
    try:
        db.session.commit()
        return jsonify({
            "message": "Perfil actualizado correctamente"
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@auth_bp.route('/profile', methods=['DELETE'])
@jwt_required()
def deactivate_profile():
    current_doctor_id = get_jwt_identity()
    doctor = Doctor.query.get(current_doctor_id)

    if not doctor:
        return jsonify({"error": "Doctor no encontrado"}), 404

    doctor.is_active = False
    
    try:
        db.session.commit()
        return jsonify({"message": "La cuenta ha sido desactivada"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500