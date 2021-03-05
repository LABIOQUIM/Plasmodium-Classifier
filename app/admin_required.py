from .routes import current_app, current_user
from functools import wraps

#atualização para evitar que a pagina quebre
def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            user = current_user.admin
        except:
            user = ""
        if user != 'True':
            return current_app.login_manager.unauthorized()
        else:
            return fn(*args, **kwargs)
    return wrapper    