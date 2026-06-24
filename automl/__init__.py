"""
automl/__init__.py
------------------
Flask application factory.
"""
from flask import Flask
from config import config


def create_app(config_name: str = "default") -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Register blueprints
    from automl.blueprints.upload import upload_bp
    from automl.blueprints.configure import configure_bp
    from automl.blueprints.training import training_bp
    from automl.blueprints.results import results_bp
    from automl.blueprints.predict import predict_bp
    from automl.blueprints.export import export_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(configure_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(export_bp)

    return app
