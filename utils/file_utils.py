import os

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'wav'}  # 根据需要添加其他允许的扩展名


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_file_in_request(request):
    if 'audio_file1' not in request.files or 'audio_file2' not in request.files or 'audio_file3' not in request.files:
        return False, 'No file part', None

    files = [request.files['audio_file1'], request.files['audio_file2'], request.files['audio_file3']]
    for file in files:
        if (file.filename == ''):
            return False, 'No selected file', None
        if not (file and allowed_file(file.filename)):
            return False, 'File type not allowed', None

    return True, '', files


def save_file(file, upload_folder):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = secure_filename(file.filename)
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)
    return file_path
