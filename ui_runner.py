# ui_runner.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
)
from PyQt5.QtCore import Qt


def process_file(file_path):
    print("处理文件：", file_path)
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"文件大小：{size} 字节")
    return "处理完成"


class FileDropApp:
    def __init__(self, process_func=process_file):
        self.app = QApplication(sys.argv)
        self.window = FileDropWidget(process_func)

    def run(self):
        self.window.show()
        sys.exit(self.app.exec_())


class FileDropWidget(QWidget):
    def __init__(self, process_func=process_file):
        super().__init__()
        self.process_func = process_func
        self.current_file = None

        self.setWindowTitle("文件选择与拖拽")
        self.resize(400, 220)

        # 初始为置顶
        self.always_on_top = True
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setAcceptDrops(True)

        # UI 元件
        self.label = QLabel("拖入文件或点击按钮选择文件", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.button_select = QPushButton("选择文件", self)
        self.button_select.clicked.connect(self.open_file_dialog)

        self.button_process = QPushButton("开始处理", self)
        self.button_process.clicked.connect(self.handle_process)

        self.toggle_button = QPushButton("🔒 取消置顶", self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle_always_on_top)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button_select)
        layout.addWidget(self.button_process)
        layout.addWidget(self.toggle_button)
        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.current_file = file_path
            self.label.setText(f"已拖入：{os.path.basename(file_path)}")

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件")
        if file_path:
            self.current_file = file_path
            self.label.setText(f"已选择：{os.path.basename(file_path)}")

    def handle_process(self):
        if self.current_file:
            result = self.process_func(self.current_file)
            print(result)
        else:
            print("尚未选择文件！")

    def toggle_always_on_top(self):
        self.always_on_top = not self.always_on_top
        flags = self.windowFlags()
        if self.always_on_top:
            self.setWindowFlags(flags | Qt.WindowStaysOnTopHint)
            self.toggle_button.setText("🔒 取消置顶")
            self.toggle_button.setChecked(True)
        else:
            self.setWindowFlags(flags & ~Qt.WindowStaysOnTopHint)
            self.toggle_button.setText("🔓 切换置顶")
            self.toggle_button.setChecked(False)
        self.show()


