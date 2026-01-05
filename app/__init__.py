from flask import Flask, send_from_directory
from .config import Config
from .extensions import db, migrate, jwt, cors
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)


    # ruta absoluta del certificado ca.pem en la raíz
    ca_path = os.path.join(os.getcwd(), "ca.pem")
    
    # argumentos de conexión SSL en SQLAlchemy
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        "connect_args": {
            "ssl": {
                "ca": ca_path
            }
        }
    }

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    
    # producción
    cors.init_app(app, resources={
        r"/api/*": {
            "origins": ["https://covid-detection-app.netlify.app", "http://localhost:3000"]
        }
    })

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