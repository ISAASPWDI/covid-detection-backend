from flask import Flask, send_from_directory
from .config import Config
from .extensions import db, migrate, jwt, cors
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}}) # cambiar en producción


    # 👇 ESTA LÍNEA ES OBLIGATORIA
    from app import models  

    MEDIA_ROOT = os.path.join(os.getcwd(), "media")

    @app.route("/media/<path:filename>")
    def media_files(filename):
        return send_from_directory(MEDIA_ROOT, filename)

    from app.routes.diagnosis_routes import diagnosis_bp
    from app.routes.auth_routes import auth_bp
    from app.routes.patient_routes import patient_bp
    from app.routes.dashboard_routes import dashboard_bp

    app.register_blueprint(diagnosis_bp, url_prefix="/api/diagnoses")
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(patient_bp, url_prefix="/api/patients")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    return app
