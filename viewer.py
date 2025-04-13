import sys
import json
import struct
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, 
                            QTreeWidgetItem, QSplitter, QTextEdit, 
                            QFileDialog, QVBoxLayout, QWidget, 
                            QHeaderView, QLabel, QHBoxLayout, QStatusBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

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
        self.tree. header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tree.setFont(QFont("Consolas", 10))
        self.tree.itemClicked.connect(self.on_item_clicked)
        self.left_layout.addWidget(self.tree)
        
        # 右侧面板
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        
        # 参数详情标签
        self.detail_label = QLabel("Parameter Details")
        self.detail_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.right_layout.addWidget(self.detail_label)
        
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
                    except:
                        info_text += "\n(Unable to load tensor data)"
                else:
                    info_text += "\n(Tensor too large to preview)"
                
                self.text_view.setText(info_text)
        else:  # 元数据项
            self.text_view.setText(f"Metadata: {item.text(0)} = {item.text(3)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SafetensorsViewer()
    viewer.show()
    sys.exit(app.exec_())