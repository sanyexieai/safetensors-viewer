import sys
import json
import struct
import numpy as np
import os
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, 
                            QTreeWidgetItem, QSplitter, QTextEdit, 
                            QFileDialog, QVBoxLayout, QWidget, 
                            QHeaderView, QLabel, QHBoxLayout, QStatusBar,
                            QPushButton, QMessageBox, QInputDialog,
                            QMenu, QAction)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from safetensors import safe_open
from safetensors.torch import save_file
import torch

class SafetensorsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Model Viewer")
        self.setGeometry(100, 100, 1000, 800)
        
        # 创建主部件
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        
        # 主布局
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)
        
        # 创建工具栏
        self.toolbar = self.addToolBar("Structure")
        self.toolbar.setMovable(False)
        self.toolbar.setStyleSheet("""
            QToolBar {
                spacing: 5px;
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 5px;
                color: #333;
            }
            QToolButton:hover {
                background-color: #e3f2fd;
            }
        """)
        
        # 添加结构编辑按钮
        self.add_tensor_action = self.toolbar.addAction("Add Tensor")
        self.add_tensor_action.triggered.connect(self.add_tensor)
        self.add_tensor_action.setEnabled(False)
        
        self.toolbar.addSeparator()
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.setStyleSheet("""
            QStatusBar {
                background-color: #f5f5f5;
                border-top: 1px solid #ddd;
                padding: 2px;
                font-size: 11px;
            }
        """)
        
        # 创建水平分割器
        self.h_splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(self.h_splitter)
        
        # 左侧面板
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)
        
        # 创建层级树形视图
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Layer/Parameter", "Shape", "Type", "Size"])
        self.tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.setFont(QFont("Consolas", 10))
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.left_layout.addWidget(self.tree)
        
        # 右侧面板
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        
        # 参数详情标签和按钮布局
        self.detail_header = QWidget()
        self.detail_header_layout = QHBoxLayout()
        self.detail_header.setLayout(self.detail_header_layout)
        
        self.detail_label = QLabel("Parameter Details")
        self.detail_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.detail_header_layout.addWidget(self.detail_label)
        
        # 添加编辑和保存按钮
        self.edit_button = QPushButton("Edit Value")
        self.edit_button.setEnabled(False)
        self.edit_button.clicked.connect(self.edit_tensor)
        self.edit_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.detail_header_layout.addWidget(self.edit_button)
        
        self.save_button = QPushButton("Save Changes")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_changes)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.detail_header_layout.addWidget(self.save_button)
        
        self.right_layout.addWidget(self.detail_header)
        
        # 创建详情视图
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setFont(QFont("Consolas", 10))
        self.right_layout.addWidget(self.text_view)
        
        # 添加面板到分割器
        self.h_splitter.addWidget(self.left_panel)
        self.h_splitter.addWidget(self.right_panel)
        
        # 设置分割比例
        self.h_splitter.setStretchFactor(0, 2)
        self.h_splitter.setStretchFactor(1, 3)
        
        # 创建菜单栏
        self.create_menu()
        
        # 初始化变量
        self.file_data = {}
        self.file_path = ""
        self.current_tensor = None
        self.modified_tensors = {}
        
        # 设置样式
        self.setup_style()
    
    def setup_style(self):
        # 设置整体样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTreeWidget {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            QTreeWidget::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #2196F3;
                padding: 5px;
            }
        """)
    
    def create_menu(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
            QMenuBar::item:selected {
                background-color: #e3f2fd;
            }
        """)
        
        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open Model")
        open_action.triggered.connect(self.open_file)
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
    
    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model File", "", "Safetensors Files (*.safetensors)"
        )
        
        if file_path:
            self.file_path = file_path
            self.load_file(file_path)
            self.add_tensor_action.setEnabled(True)  # 启用添加张量按钮
    
    def organize_tensor_tree(self, tensors):
        # 组织层级结构
        layers = {}
        for key, info in tensors.items():
            if "." in key:
                layer_name, param_name = key.rsplit(".", 1)
                if layer_name not in layers:
                    layers[layer_name] = {}
                layers[layer_name][param_name] = info
            else:
                if "root" not in layers:
                    layers["root"] = {}
                layers["root"][key] = info
        return layers
    
    def load_file(self, file_path):
        self.tree.clear()
        self.text_view.clear()
        
        try:
            with open(file_path, "rb") as f:
                length_bytes = f.read(8)
                header_length = struct.unpack("<Q", length_bytes)[0]
                
                header_bytes = f.read(header_length)
                header = json.loads(header_bytes.decode("utf-8"))
                
                metadata = header.get("__metadata__", {})
                self.file_data = {"metadata": metadata, "tensors": {}}
                
                # 添加元数据
                if metadata:
                    metadata_item = QTreeWidgetItem(["Metadata", "", "dict", ""])
                    metadata_item.setBackground(0, QColor("#e3f2fd"))
                    self.tree.addTopLevelItem(metadata_item)
                    
                    for key, value in metadata.items():
                        metadata_item.addChild(QTreeWidgetItem([key, "", str(type(value)), str(value)]))
                
                # 组织并添加张量信息
                tensors = {}
                for key, tensor_info in header.items():
                    if key != "__metadata__":
                        dtype = tensor_info["dtype"]
                        shape = tensor_info["shape"]
                        data_offsets = tensor_info["data_offsets"]
                        size = data_offsets[1] - data_offsets[0]
                        
                        tensors[key] = {
                            "dtype": dtype,
                            "shape": shape,
                            "size": size,
                            "offsets": data_offsets
                        }
                
                # 组织层级结构
                layers = self.organize_tensor_tree(tensors)
                
                # 添加到树形视图
                for layer_name, params in layers.items():
                    layer_item = QTreeWidgetItem([layer_name, "", "", ""])
                    layer_item.setBackground(0, QColor("#e3f2fd"))
                    self.tree.addTopLevelItem(layer_item)
                    
                    for param_name, info in params.items():
                        param_item = QTreeWidgetItem([
                            param_name,
                            str(info["shape"]),
                            info["dtype"],
                            f"{info['size']} bytes"
                        ])
                        layer_item.addChild(param_item)
                        
                        # 存储张量信息
                        full_name = f"{layer_name}.{param_name}" if layer_name != "root" else param_name
                        self.file_data["tensors"][full_name] = info
                
                # 展开所有项
                self.tree.expandAll()
                
                # 更新模型信息
                self.statusBar.showMessage(f"Model Structure - {file_path.split('/')[-1]}")
                
        except Exception as e:
            self.text_view.setText(f"Error loading file: {str(e)}")
    
    def edit_tensor(self):
        if not self.current_tensor:
            return
            
        try:
            with safe_open(self.file_path, framework="pt") as f:
                tensor = f.get_tensor(self.current_tensor)
                
            # 获取当前值
            current_value = tensor.flatten().tolist()
            if len(current_value) > 100:
                QMessageBox.warning(self, "Warning", 
                    "This tensor is too large to edit directly. Please use a script for bulk modifications.")
                return
                
            # 显示编辑对话框
            text, ok = QInputDialog.getText(self, 'Edit Tensor Values',
                'Enter new values (comma-separated):',
                text=','.join(map(str, current_value)))
                
            if ok:
                try:
                    # 解析新值
                    new_values = [float(x.strip()) for x in text.split(',')]
                    if len(new_values) != len(current_value):
                        raise ValueError("Number of values must match tensor size")
                        
                    # 创建新张量
                    new_tensor = np.array(new_values).reshape(tensor.shape)
                    
                    # 存储修改后的张量
                    self.modified_tensors[self.current_tensor] = new_tensor
                    
                    # 更新显示
                    self.save_button.setEnabled(True)
                    self.text_view.append("\nModified values (not saved):\n" + str(new_tensor))
                    self.statusBar.showMessage("Tensor modified. Click 'Save Changes' to apply.")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Invalid input: {str(e)}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to edit tensor: {str(e)}")

    def save_changes(self):
        if not self.modified_tensors:
            return
            
        try:
            # 读取所有现有张量
            tensors = {}
            with safe_open(self.file_path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            
            # 更新修改的张量
            for key, tensor in self.modified_tensors.items():
                tensors[key] = tensor
            
            # 创建备份
            backup_path = self.file_path + ".backup"
            if not os.path.exists(backup_path):
                shutil.copy2(self.file_path, backup_path)
            
            # 保存修改后的文件
            save_file(tensors, self.file_path)
            
            # 清除修改记录
            self.modified_tensors.clear()
            self.save_button.setEnabled(False)
            
            QMessageBox.information(self, "Success", 
                f"Changes saved successfully.\nBackup created at: {backup_path}")
            
            # 重新加载文件
            self.load_file(self.file_path)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save changes: {str(e)}")

    def on_item_clicked(self, item, column):
        parent = item.parent()
        if parent is None:  # 层级项
            info_text = f"Layer: {item.text(0)}\n"
            info_text += "-" * 50 + "\n"
            child_count = item.childCount()
            info_text += f"Parameters count: {child_count}\n"
            total_size = 0
            for i in range(child_count):
                child = item.child(i)
                param_size = int(child.text(3).split()[0])
                total_size += param_size
            info_text += f"Total size: {total_size:,} bytes"
            self.text_view.setText(info_text)
            self.edit_button.setEnabled(False)
            self.current_tensor = None
            
        elif parent.text(0) != "Metadata":  # 参数项
            layer_name = parent.text(0)
            param_name = item.text(0)
            full_name = f"{layer_name}.{param_name}" if layer_name != "root" else param_name
            tensor_info = self.file_data["tensors"].get(full_name)
            
            if tensor_info:
                info_text = f"Parameter: {param_name}\n"
                info_text += "-" * 50 + "\n"
                info_text += f"Layer: {layer_name}\n"
                info_text += f"Type: {tensor_info['dtype']}\n"
                info_text += f"Shape: {tensor_info['shape']}\n"
                info_text += f"Size: {tensor_info['size']:,} bytes\n"
                
                # 尝试加载实际张量数据
                if np.prod(tensor_info["shape"]) <= 100:
                    try:
                        from safetensors import safe_open
                        with safe_open(self.file_path, framework="pt") as f:
                            tensor = f.get_tensor(full_name)
                            info_text += f"\nValue Preview:\n{tensor}"
                            # 允许编辑小张量
                            self.edit_button.setEnabled(True)
                            self.current_tensor = full_name
                    except:
                        info_text += "\n(Unable to load tensor data)"
                        self.edit_button.setEnabled(False)
                        self.current_tensor = None
                else:
                    info_text += "\n(Tensor too large to preview)"
                    self.edit_button.setEnabled(False)
                    self.current_tensor = None
                
                self.text_view.setText(info_text)
        else:  # 元数据项
            self.text_view.setText(f"Metadata: {item.text(0)} = {item.text(3)}")
            self.edit_button.setEnabled(False)
            self.current_tensor = None

    def contextMenuEvent(self, event):
        # 获取树形视图中的点击位置
        pos = self.tree.mapFromGlobal(event.globalPos())
        item = self.tree.itemAt(pos)
        
        if item and item.parent():  # 只对参数项显示右键菜单
            menu = QMenu(self)
            
            rename_action = QAction("Rename", self)
            rename_action.triggered.connect(lambda: self.rename_tensor(item))
            menu.addAction(rename_action)
            
            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self.delete_tensor(item))
            menu.addAction(delete_action)
            
            menu.exec_(event.globalPos())

    def add_tensor(self):
        try:
            # 获取新张量名称
            name, ok = QInputDialog.getText(self, 'Add New Tensor', 'Enter tensor name:')
            if not ok or not name:
                return
                
            # 获取张量形状
            shape_str, ok = QInputDialog.getText(self, 'Tensor Shape', 
                'Enter shape (comma-separated, e.g., 3,224,224):')
            if not ok or not shape_str:
                return
                
            try:
                shape = tuple(map(int, shape_str.split(',')))
            except:
                QMessageBox.critical(self, "Error", "Invalid shape format")
                return
                
            # 创建新张量
            try:
                # 默认创建零张量
                new_tensor = torch.zeros(shape)
                
                # 读取所有现有张量
                tensors = {}
                with safe_open(self.file_path, framework="pt") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                
                # 添加新张量
                tensors[name] = new_tensor
                
                # 创建备份
                backup_path = self.file_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(self.file_path, backup_path)
                
                # 保存修改后的文件
                save_file(tensors, self.file_path)
                
                # 重新加载文件
                self.load_file(self.file_path)
                
                QMessageBox.information(self, "Success", 
                    f"New tensor '{name}' added successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to add tensor: {str(e)}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add tensor: {str(e)}")

    def rename_tensor(self, item):
        if not item or not item.parent():
            return
            
        old_name = item.text(0)
        layer_name = item.parent().text(0)
        full_old_name = f"{layer_name}.{old_name}" if layer_name != "root" else old_name
        
        # 获取新名称
        new_name, ok = QInputDialog.getText(self, 'Rename Tensor', 
            'Enter new name:', text=old_name)
        
        if ok and new_name and new_name != old_name:
            try:
                # 读取所有张量
                tensors = {}
                with safe_open(self.file_path, framework="pt") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                
                # 重命名张量
                if full_old_name in tensors:
                    tensors[new_name] = tensors.pop(full_old_name)
                    
                    # 创建备份
                    backup_path = self.file_path + ".backup"
                    if not os.path.exists(backup_path):
                        shutil.copy2(self.file_path, backup_path)
                    
                    # 保存修改后的文件
                    save_file(tensors, self.file_path)
                    
                    # 重新加载文件
                    self.load_file(self.file_path)
                    
                    QMessageBox.information(self, "Success", 
                        f"Tensor renamed from '{old_name}' to '{new_name}'")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to rename tensor: {str(e)}")

    def delete_tensor(self, item):
        if not item or not item.parent():
            return
            
        name = item.text(0)
        layer_name = item.parent().text(0)
        full_name = f"{layer_name}.{name}" if layer_name != "root" else name
        
        reply = QMessageBox.question(self, 'Delete Tensor',
            f"Are you sure you want to delete tensor '{name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # 读取所有张量
                tensors = {}
                with safe_open(self.file_path, framework="pt") as f:
                    for key in f.keys():
                        if key != full_name:  # 排除要删除的张量
                            tensors[key] = f.get_tensor(key)
                
                # 创建备份
                backup_path = self.file_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(self.file_path, backup_path)
                
                # 保存修改后的文件
                save_file(tensors, self.file_path)
                
                # 重新加载文件
                self.load_file(self.file_path)
                
                QMessageBox.information(self, "Success", 
                    f"Tensor '{name}' deleted successfully")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete tensor: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SafetensorsViewer()
    viewer.show()
    sys.exit(app.exec_())