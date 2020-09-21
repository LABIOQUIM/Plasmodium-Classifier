from app import app, login_manager, db
from flask import render_template, request, redirect, url_for, flash, send_file, current_app
from flask_login import logout_user, login_required, login_user, current_user
from .models import User
from .config import os, Config
from .admin_required import admin_required
from .upload_img import upload_file
from .classifier import classify_image

@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        name = request.form.get('name')
        user = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        passconfirm = request.form.get('passwordconfirm')
        #faz checagem para verificar se usuário ou senha já são utilizados    
        check_email = User.query.filter(User.email == email).first()
        check_user = User.query.filter(User.username == user).first()
        if check_email is None and check_user is None:
            new = User(name=name,username=user,email=email,register='False')
            new.set_password(password)
            db.session.add(new)
            db.session.commit()
            flash('Solicitação de cadastro do(a) Usuário(a) {} realizada com sucesso. Em breve responderemos por Email se a solicitação foi aceita.'.format(user), 'primary')
            return redirect(url_for('login'))
        else:
            flash('Erro, email  ou usuário já estão sendo utilizados.', 'danger')
            return redirect(url_for('cadastro'))

    flash('Por favor, preencha os dados corretamente. Em caso de dados incorretos a solicitação de cadastro será cancelada.', 'danger')
    return render_template('cadastro.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        form_entry = request.form.get('username')
        user = User.query.filter((User.username == form_entry) | (User.email == form_entry)).first()
        #verifica se o usuário existe
        if user is None or not user.check_password(request.form.get('password')):
            flash('Usuário ou senha inválidos', 'danger')
            return render_template('login.html')
        #verifica se o cadastro do usuário é aceito.
        if user.register == 'False':
            flash('Seu cadastro ainda não foi aceito, aguarde o Email de confirmação.', 'danger')     
        else :
            login_user(user)
            return redirect(url_for('protected'))
    return render_template('login.html')

@app.route('/protected')
@login_required
def protected():
    flash('Olá {}, seja bem-vindo(a)'.format(current_user.username), 'primary')
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    show_result = False
    if request.method == 'POST':
        show_result = True
        file = request.files.get('file')
        upload_file(file)
        img = '/static/img/upload/' + file.filename
        result = classify_image(file)
        
        return render_template('index.html', actindex = 'active', show_result=show_result, img=img,
            result=result['classification'], probability=result['probability'], _class=result['class'])

    return render_template('index.html', actindex = 'active', show_result=show_result)
    
    
