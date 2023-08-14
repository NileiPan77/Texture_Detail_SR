from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QLabel, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import sys
import numpy as np
import torch
import pyexr
from fourier_filtering import *
# Fourier filtering function here...

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.title = "Fourier Filtering GUI"
        self.top = 100
        self.left = 100
        self.width = 800
        self.height = 600

        self.target_exr = None
        self.base_exr = None

        self.init_window()

    def init_window(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)

        # Set layout
        self.layout = QVBoxLayout()
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(self.layout)

        # Add button for loading target
        self.load_target_button = QPushButton('Load Target', self)
        self.load_target_button.clicked.connect(self.load_target)
        self.layout.addWidget(self.load_target_button)

        # Add button for loading base
        self.load_base_button = QPushButton('Load Base', self)
        self.load_base_button.clicked.connect(self.load_base)
        self.layout.addWidget(self.load_base_button)

        # Add slider for 'r' parameter
        self.r_slider = QSlider(Qt.Horizontal, self)
        self.r_slider.setRange(1, 100)
        self.r_slider.valueChanged.connect(self.run_fourier_filter)
        self.layout.addWidget(self.r_slider)

        # Add slider for 'r_high' parameter
        self.r_high_slider = QSlider(Qt.Horizontal, self)
        self.r_high_slider.setRange(1, 100)
        self.r_high_slider.valueChanged.connect(self.run_fourier_filter)
        self.layout.addWidget(self.r_high_slider)

        # Add slider for 'degree' parameter
        self.degree_slider = QSlider(Qt.Horizontal, self)
        self.degree_slider.setRange(1, 100)
        self.degree_slider.valueChanged.connect(self.run_fourier_filter)
        self.layout.addWidget(self.degree_slider)

        # Add labels for image display
        self.label_high = QLabel(self)
        self.layout.addWidget(self.label_high)

        self.label_low = QLabel(self)
        self.layout.addWidget(self.label_low)

        self.label_combined = QLabel(self)
        self.layout.addWidget(self.label_combined)

        self.show()

    def load_target(self):
        filename, _ = QFileDialog.getOpenFileName()
        if filename:
            self.target_exr = np.array(pyexr.open(filename))

    def load_base(self):
        filename, _ = QFileDialog.getOpenFileName()
        if filename:
            self.base_exr = np.array(pyexr.open(filename))

    def run_fourier_filter(self):
        if self.target_exr is None or self.base_exr is None:
            return

        r = self.r_slider.value()
        r_high = self.r_high_slider.value()
        degree = self.degree_slider.value()

        # Run Fourier filtering
        image_high, image_low, image_combined = fourier_filtering(self.target_exr, self.base_exr, r, r_high, degree)

        # Display images
        self.display_image(image_high, self.label_high)
        self.display_image(image_low, self.label_low)
        self.display_image(image_combined, self.label_combined)

    def display_image(self, img, label):
        # Convert the NumPy array to a QImage
        height, width = img.shape[:2]
        bytes_per_line = width * 3
        qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Set the QPixmap for the label
        label.setPixmap(QPixmap.fromImage(qimage))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
