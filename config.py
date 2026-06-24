import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "automl-studio-secret-2024")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    SESSION_FOLDER = os.path.join(BASE_DIR, "sessions")
    MODEL_FOLDER = os.path.join(BASE_DIR, "saved_models")
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB upload limit
    ALLOWED_EXTENSIONS = {"csv"}
    
    # Required for Hugging Face Spaces iframe embedding
    SESSION_COOKIE_SAMESITE = "None"
    SESSION_COOKIE_SECURE = True

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.SESSION_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
