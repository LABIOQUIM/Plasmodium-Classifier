from app import app, login_manager, db
from flask import render_template, request, redirect, url_for, flash, send_file, current_app
from flask_login import logout_user, login_required, login_user, current_user
from .models import User
from .config import os, Config
from .admin_required import admin_required
from .upload_img import upload_file
from .classifier import classify_image, training

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
            flash('Solicitação de cadastro do(a) Usuário(a) {} realizada com sucesso. Em breve seu cadastro será ativado.'.format(user), 'primary')
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
            flash('Seu cadastro ainda não foi aceito, aguarde a confirmação.', 'danger')     
        else :
            login_user(user)
            return redirect(url_for('protected'))
    return render_template('login.html')

@app.route('/protected')
@login_required
def protected():
    flash('Olá {}, seja bem-vindo(a)'.format(current_user.username), 'primary')
    return redirect(url_for('index'))

@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect(url_for('logout'))

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
        result = classify_image(file)
        img =  '/static/img/upload/' + file.filename
        
        return render_template('index.html', actindex = 'active', show_result=show_result, img=img,
            result=result['classification'], probability=result['probability'], _class=result['class'])

    return render_template('index.html', actindex = 'active', show_result=show_result)

@app.route('/admin', methods=['GET', 'POST'])
@admin_required
def admin():
    UserData = User.query.filter(User.register == 'True')
    return render_template('admin.html', actadmin = 'active', UserData=UserData)

@app.route('/admin/edit/<int:id>', methods=['GET', 'POST'])
@admin_required
def edit_user(id):
    if request.method == 'POST':
        name = request.form.get('name')
        user = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        passconfirm = request.form.get('passwordconfirm')
        if password == '' and passconfirm == '':
            UserData = User.query.get(int(id))
            UserData.name = name
            UserData.username = user
            UserData.email = email
            try:
                db.session.add(UserData)
                db.session.commit()
                flash('Dados do(a) usuário(a) {} alterados com sucesso.'.format(user), 'primary')
                return redirect(url_for('admin'))
            except:
                flash('Erro, email  ou usuário já estão sendo utilizados.', 'danger')
                return redirect(url_for('edit_user', id=id))

        elif password == passconfirm:
            UserData = User.query.get(int(id))
            UserData.name = name
            UserData.username = user
            UserData.email = email
            try:
                UserData.set_password(password)
                db.session.add(UserData)
                db.session.commit()
                flash('Dados do(a) usuário(a) {} alterados com sucesso.'.format(user), 'primary')
                return redirect(url_for('admin'))
            except:
                flash('Erro, email  ou usuário já estão sendo utilizados.', 'danger')
                return redirect(url_for('edit_user', id=id))

        flash('Erro ao editar usuário(a) {}.'.format(user), 'danger')
        return redirect(url_for('admin'))
    UserData = User.query.get(int(id))
    return render_template('edit_user.html', UserData=UserData)


@app.route('/admin/newUser', methods=['GET', 'POST'])
@admin_required
def newuser():
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
            new = User(name=name,username=user,email=email,register='True')
            new.set_password(password)
            db.session.add(new)
            db.session.commit()
            flash('Cadastro do(a) Usuário(a) {} realizado com sucesso.'.format(user), 'primary')
            return redirect(url_for('admin'))
        else:
            flash('Erro, email  ou usuário já estão sendo utilizados.', 'danger')
            return redirect(url_for('newuser'))
  
    return render_template('new_user.html')

@app.route('/admin/cadastros', methods=['GET', 'POST'])
@admin_required
def admin_cadastros():
    NewUserData = User.query.filter(User.register == 'False')
    return render_template('admin_cadastros.html', NewUserData=NewUserData)

@app.route('/admin/accept_newUser/<int:id>', methods=['GET', 'POST'])
@admin_required
def accept_newUser(id):
    #ativa o cadastro do usuário.
    UserData = User.query.get(int(id))
    UserData.register = 'True'
    name = UserData.name
    email = UserData.email
    db.session.add(UserData)
    db.session.commit()
    flash('Solicitação de cadastro do(a) usuário(a) {} aceita com sucesso.'.format(UserData.username), 'primary')
    return redirect(url_for('admin_cadastros'))

@app.route('/admin/remove_newUser/<int:id>')
@admin_required
def remove_newUser(id):
    UserData = User.query.get(int(id))
    name = UserData.name
    email = UserData.email
    db.session.delete(UserData)
    db.session.commit()   
    flash('Solicitação de cadastro do(a) usuário(a) {} removida com sucesso.'.format(UserData.username), 'primary')
    return redirect(url_for('admin_cadastros'))


@app.route('/admin/remove/<int:id>')
@admin_required
def removeuser(id):
    UserData = User.query.get(int(id))
    if UserData.username != 'admin':
        db.session.delete(UserData)
        db.session.commit()
        flash('Usuário(a) {} removido(a) com sucesso.'.format(UserData.username), 'primary')
        return redirect(url_for('admin'))
    flash('Não é possível remover o admin', 'danger')
    return redirect(url_for('admin'))

@app.route('/admin/training', methods=['GET', 'POST'])
@admin_required
def execute_training():
    training()
    return redirect(url_for('admin'))