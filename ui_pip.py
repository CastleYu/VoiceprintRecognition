import sys
from urllib.request import getproxies

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QMessageBox, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QDialog, QTextEdit, QMenuBar, QAction, QFormLayout, QFileDialog, QCheckBox
)
from PyQt5.QtCore import QProcess, Qt


class ConsoleDialog(QDialog):
    def __init__(self, title="命令输出"):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.output_area)
        self.setLayout(layout)

    def append_output(self, text):
        self.output_area.append(text)


class PipManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Pip包管理程序（增强版）")
        self.setGeometry(400, 200, 520, 1100)
        self.process = None
        self.full_package_data = []
        self.python_path = sys.executable
        self.pip_proxy = self.get_current_pip_proxy()
        self.use_system_proxy = False
        self.system_proxy = ""
        self.default_index_url = self.get_current_pip_index_url()
        self.temp_index_url = self.default_index_url
        self.initUI()
        self.load_packages()

    def initUI(self):
        layout = QVBoxLayout()

        python_version_label = QLabel(f"当前Python版本：{sys.version.split()[0]}")
        layout.addWidget(python_version_label)

        menu_bar = QMenuBar(self)
        settings_menu = menu_bar.addMenu("设置")
        open_settings_action = QAction("打开设置", self)
        open_settings_action.triggered.connect(self.open_settings_dialog)
        settings_menu.addAction(open_settings_action)
        layout.setMenuBar(menu_bar)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索包名...")
        self.search_input.textChanged.connect(self.filter_packages)
        upgrade_btn = QPushButton("升级 pip")
        upgrade_btn.clicked.connect(self.upgrade_pip)
        search_layout.addWidget(upgrade_btn)
        search_layout.addWidget(QLabel("🔍搜索："))
        search_layout.addWidget(self.search_input)

        temp_mirror_layout = QHBoxLayout()
        self.temp_index_input = QLineEdit(self.temp_index_url or "")
        temp_mirror_layout.addWidget(QLabel("临时镜像源："))
        temp_mirror_layout.addWidget(self.temp_index_input)

        self.cache_checkbox = QCheckBox("使用缓存安装")
        self.cache_checkbox.setChecked(True)
        temp_mirror_layout.addWidget(self.cache_checkbox)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["包名", "版本"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(self.table.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        install_layout = QHBoxLayout()
        self.install_input = QLineEdit()
        self.install_input.setPlaceholderText("输入要安装/升级的包名（可含版本），如 requests==2.26.0 或 pandas>=1.3")
        self.install_input.returnPressed.connect(self.install_package)
        install_btn = QPushButton("安装/升级包")
        install_btn.clicked.connect(self.install_package)
        reinstall_btn = QPushButton("重新安装包")
        reinstall_btn.clicked.connect(self.reinstall_package)
        install_layout.addWidget(self.install_input)
        install_layout.addWidget(install_btn)
        install_layout.addWidget(reinstall_btn)

        btn_layout = QHBoxLayout()
        uninstall_btn = QPushButton("卸载选中包")
        uninstall_btn.clicked.connect(self.uninstall_package)
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.load_packages)
        btn_layout.addWidget(uninstall_btn)
        btn_layout.addWidget(refresh_btn)

        layout.addLayout(search_layout)
        layout.addWidget(self.table)
        layout.addLayout(install_layout)
        layout.addLayout(temp_mirror_layout)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_system_proxy(self):
        proxies = getproxies()
        return proxies.get("https") or proxies.get("http") or ""

    def get_current_pip_proxy(self):
        process = QProcess()
        process.start(sys.executable, ["-m", "pip", "config", "get", "global.proxy"])
        process.waitForFinished()
        output = process.readAllStandardOutput().data().decode().strip()
        return output if output else None

    def get_current_pip_index_url(self):
        process = QProcess()
        process.start(sys.executable, ["-m", "pip", "config", "get", "global.index-url"])
        process.waitForFinished()
        output = process.readAllStandardOutput().data().decode().strip()
        return output if output else None

    def open_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("设置")
        dialog.resize(900, 200)
        layout = QFormLayout()

        interpreter_input = QLineEdit(self.python_path)
        browse_btn = QPushButton("选择文件")

        def choose_interpreter():
            path, _ = QFileDialog.getOpenFileName(self, "选择Python解释器", "", "Python可执行文件 (*.exe)")
            if path:
                interpreter_input.setText(path)

        browse_btn.clicked.connect(choose_interpreter)

        interpreter_layout = QHBoxLayout()
        interpreter_layout.addWidget(interpreter_input)
        interpreter_layout.addWidget(browse_btn)
        layout.addRow("Python解释器：", interpreter_layout)

        proxy_input = QLineEdit(self.pip_proxy or "")
        system_proxy_check = QCheckBox("使用系统代理")
        system_proxy_check.setChecked(self.use_system_proxy)

        def use_proxy_from_system():
            self.system_proxy = self.get_system_proxy()
            proxy_input.setText(self.system_proxy)
            proxy_input.setEnabled(False)

        def toggle_proxy_editable():
            if system_proxy_check.isChecked():
                use_proxy_from_system()
            else:
                proxy_input.setEnabled(True)
                proxy_input.setText(self.pip_proxy or "")

        system_proxy_check.stateChanged.connect(toggle_proxy_editable)
        toggle_proxy_editable()

        index_input = QLineEdit(self.default_index_url or "")

        layout.addRow("全局镜像源：", index_input)
        layout.addRow("PIP代理：", proxy_input)
        layout.addRow("", system_proxy_check)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("保存")
        cancel_btn = QPushButton("取消")
        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        layout.addRow("", button_layout)

        def save_settings():
            self.python_path = interpreter_input.text().strip()
            self.use_system_proxy = system_proxy_check.isChecked()
            proxy_value = self.system_proxy if self.use_system_proxy else (proxy_input.text().strip() or None)
            index_value = index_input.text().strip()
            self.pip_proxy = proxy_value
            if index_value and index_value != self.default_index_url:
                self.run_pip_command(["config", "set", "global.index-url", index_value], "设置全局镜像源")
                self.default_index_url = index_value
                self.temp_index_input.setText(index_value)
            if proxy_value and proxy_value != self.get_current_pip_proxy():
                self.run_pip_command(["config", "set", "global.proxy", proxy_value], "设置全局代理")
            dialog.accept()

        save_btn.clicked.connect(save_settings)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.setLayout(layout)
        dialog.exec_()

    def on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.console.append_output(data)

    def on_stderr(self):
        data = self.process.readAllStandardError().data().decode()
        self.console.append_output(data)

    def on_process_finished(self):
        self.console.append_output("\n✅ 操作完成！")
        self.load_packages()

    def load_packages(self):
        self.full_package_data.clear()
        self.table.setRowCount(0)
        process = QProcess()
        process.start(sys.executable, ["-m", "pip", "list", "--format=freeze"])
        process.waitForFinished()
        output = process.readAllStandardOutput().data().decode()
        for line in output.strip().split("\n"):
            if "==" in line:
                name, version = line.strip().split("==")
                self.full_package_data.append((name, version))
        self.display_packages(self.full_package_data)

    def run_pip_command(self, args, title="正在执行命令"):
        self.console = ConsoleDialog(title)
        self.console.show()
        self.process = QProcess(self)
        env = self.process.processEnvironment()
        if self.pip_proxy:
            env.insert("https_proxy", self.pip_proxy)
            env.insert("http_proxy", self.pip_proxy)
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self.on_stdout)
        self.process.readyReadStandardError.connect(self.on_stderr)
        self.process.finished.connect(self.on_process_finished)
        self.process.start(self.python_path, ["-m", "pip"] + args)

    def display_packages(self, packages):
        self.table.setRowCount(0)
        for name, version in packages:
            row_pos = self.table.rowCount()
            self.table.insertRow(row_pos)
            self.table.setItem(row_pos, 0, QTableWidgetItem(name))
            self.table.setItem(row_pos, 1, QTableWidgetItem(version))

    def filter_packages(self):
        keyword = self.search_input.text().strip().lower()
        filtered = [pkg for pkg in self.full_package_data if keyword in pkg[0].lower()]
        self.display_packages(filtered)

    def install_package(self):
        package = self.install_input.text().strip()
        if not package:
            QMessageBox.warning(self, "警告", "请输入要安装的包名！")
            return
        args = ["install", package]
        if not self.cache_checkbox.isChecked():
            args.append("--no-cache-dir")
        index_url = self.temp_index_input.text().strip()
        if index_url:
            args += ["-i", index_url]
        self.run_pip_command(args, f"安装/升级包：{package}")

    def reinstall_package(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个包")
            return
        row = selected[0].row()
        package = self.table.item(row, 0).text()
        args = ["install", "--force-reinstall", "--no-cache-dir", package]
        index_url = self.temp_index_input.text().strip()
        if index_url:
            args += ["-i", index_url]
        self.run_pip_command(args, f"重新安装包：{package}")

    def uninstall_package(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "警告", "请先选择一个包")
            return
        row = selected[0].row()
        pkg_name = self.table.item(row, 0).text()
        confirm = QMessageBox.question(self, "确认卸载", f"确定要卸载 {pkg_name} 吗？",
                                       QMessageBox.Yes | QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.run_pip_command(["uninstall", "-y", pkg_name], f"卸载包：{pkg_name}")

    def upgrade_pip(self):
        self.run_pip_command(["install", "--upgrade", "pip"], "升级 pip")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PipManager()
    win.show()
    sys.exit(app.exec_())
