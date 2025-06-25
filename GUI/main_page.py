import sys
import os

sys.path.append("MUFASA_Fluorescence_Fluctuations_Simulation/")

from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QPushButton, QVBoxLayout, QLabel, QStackedWidget
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from GUI.simulate_one_molecule_page import SimulateOneMoleculePage
from GUI.simulate_an_image import SimulateStructurePage
from GUI.parameters_predictor import ParametersPredictorPage
from GUI.laser_range import LaserRangePage

class MainPage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()

        self.stacked_widget = stacked_widget

        # Set up the main page layout
        layout = QVBoxLayout()  # Use QVBoxLayout to stack image and buttons vertically

        # Create QLabel to display the image
        self.image_label = QLabel(self)
        # Resize the image using the scaled method
        pixmap = QPixmap("GUI/logo2.png")  # Replace with the actual path to your image
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)  # Adjust 200x200 to your desired size


        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)  # Center the image

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()  # Use QHBoxLayout for side-by-side buttons

        self.simulate_one_molecule_button = QPushButton("Simulate One Molecule")
        self.simulate_structure_button = QPushButton("Simulate a Structure")
        self.parameters_predictor_button = QPushButton("Parameters Predictor") 
        self.range_of_laser_button = QPushButton("Range of Laser") 

        # Add buttons to the horizontal layout
        button_layout.addWidget(self.simulate_one_molecule_button)
        button_layout.addWidget(self.simulate_structure_button)
        button_layout.addWidget(self.parameters_predictor_button)  
        button_layout.addWidget(self.range_of_laser_button)  

        # Add stretch to push image and buttons to the center of the window
        layout.addStretch(1)  # Add stretch to push everything down
        layout.addWidget(self.image_label)  # Add image in the middle
        layout.addLayout(button_layout)     # Add buttons directly below the image
        layout.addStretch(1)  # Add stretch to center the image and buttons in the middle


        # Connect the buttons to their respective functions
        self.simulate_one_molecule_button.clicked.connect(self.show_simulate_one_molecule_page)
        self.simulate_structure_button.clicked.connect(self.show_simulate_structure_page)
        self.parameters_predictor_button.clicked.connect(self.show_parameters_predictor_page) 
        self.range_of_laser_button.clicked.connect(self.show_range_of_laser_page)

        # Set the main layout for the widget
        self.setLayout(layout)

    def show_simulate_one_molecule_page(self):
        self.stacked_widget.setCurrentIndex(1)

    def show_simulate_structure_page(self):
        self.stacked_widget.setCurrentIndex(2)

    def show_parameters_predictor_page(self):
        self.stacked_widget.setCurrentIndex(3)
    
    def show_range_of_laser_page(self):
        self.stacked_widget.setCurrentIndex(4)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Simulation Options")
        self.setGeometry(100, 100, 800, 800)

        # Create a stacked widget for page navigation
        self.stacked_widget = QStackedWidget()

        self.main_page = MainPage(self.stacked_widget)
        self.simulate_one_molecule_page = SimulateOneMoleculePage(self.stacked_widget)
        self.simulate_structure_page = SimulateStructurePage(self.stacked_widget)
        self.parameters_predictor_page = ParametersPredictorPage(self.stacked_widget) 
        self.laser_range_page = LaserRangePage(self.stacked_widget)

        # Add pages to stacked widget
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.simulate_one_molecule_page)
        self.stacked_widget.addWidget(self.simulate_structure_page)
        self.stacked_widget.addWidget(self.parameters_predictor_page)  
        self.stacked_widget.addWidget(self.laser_range_page)

        # Set the layout to stacked widget
        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)
        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    dark_stylesheet = """
QWidget {
    background-color: #000000;  /* Black background */
    color: #f0f0f0;  /* Light grey text color */
}
QGroupBox {
    border: 1px solid #4a4a4a;  /* Dark grey border */
    border-radius: 5px;
    margin-top: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 5px;
    color: #f0f0f0;  /* Light grey text color for GroupBox title */
}
QLineEdit, QComboBox, QPushButton {
    background-color: #1a1a1a;  /* Dark grey background for inputs */
    border: 1px solid #666666;  /* Grey border */
    padding: 5px;
    border-radius: 3px;
    color: #f0f0f0;  /* Light grey text color */
}
QPushButton {
    background-color: #333333;  /* Slightly lighter grey for buttons */
    border: 1px solid #888888;  /* Light grey border */
    border-radius: 5px;
    padding: 5px 10px;
    color: #f0f0f0;  /* Light grey text color */
}
QPushButton:hover {
    background-color: #444444;  /* Medium grey when hovered */
}
"""

    app.setStyleSheet(dark_stylesheet)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
