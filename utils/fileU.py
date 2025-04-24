import os

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'wav'}  # 根据需要添加其他允许的扩展名


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_file_in_request(request):
    files = []
    index = 1
    while f'audio_file{index}' in request.files:
        file = request.files[f'audio_file{index}']
        if file.filename == '':
            return False, f'No selected file for audio_file{index}', None
        if not allowed_file(file.filename):
            return False, f'File type not allowed for audio_file{index}', None
        files.append(file)
        index += 1

    if not files:
        return False, 'No file parts', None

    return True, '', files


def save_file(file, upload_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path


def create_path(*path):
    return os.path.abspath(os.path.join(*path))



