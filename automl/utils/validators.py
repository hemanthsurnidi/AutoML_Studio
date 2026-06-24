import os
from config import Config


def allowed_file(filename: str) -> bool:
    """Check if uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def validate_csv_upload(file) -> tuple[bool, str]:
    """
    Validate an uploaded file object.
    Returns (is_valid: bool, error_message: str).
    """
    if not file or file.filename == "":
        return False, "No file selected."

    if not allowed_file(file.filename):
        return False, "Only CSV files are allowed. Please upload a .csv file."

    return True, ""
