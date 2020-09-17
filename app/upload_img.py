from .config import Config
from werkzeug.utils import secure_filename
import os

basedir = os.path.abspath(os.path.dirname(__file__))

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def upload_file(file):
    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(basedir, 'static', 'img', 'upload', filename))
        return True
    else:
        return False
