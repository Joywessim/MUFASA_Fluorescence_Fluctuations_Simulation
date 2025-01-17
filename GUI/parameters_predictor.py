import sys
import os

sys.path.append("Fluorescence_Fluctuations_Simulation/")
# Ensure Matplotlib uses PySide6
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')
import numpy as np
from PySide6.QtWidgets import QFileDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d

class ParametersPredictorPage(QWidget):
    def __init__(self, stacked_widget=None):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.photon_signal = None
        self.denoised_signal = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        form_layout = QFormLayout()

        # File upload button
        self.upload_button = QPushButton("Upload Photon Signal (.npy)")
        self.upload_button.clicked.connect(self.upload_file)
        form_layout.addRow("Photon Signal:", self.upload_button)

        # Parameter selection (Epsilon or Excitation Power)
        self.param_selector = QComboBox()
        self.param_selector.addItems(["Epsilon", "Excitation Power"])
        form_layout.addRow("Select Parameter to Estimate:", self.param_selector)

        # Input fields for other parameters
        self.exc_wavelength_input = QLineEdit()
        self.power_input = QLineEdit()
        self.epsilon_input = QLineEdit()
        self.N_input = QLineEdit("6.022e23")  # Avogadro's number
        self.h_input = QLineEdit("6.626e-34")  # Planck's constant
        self.c_input = QLineEdit("3e8")  # Speed of light in vacuum (m/s)
        self.frame_length_input = QLineEdit("1e-2")  # Default frame length input

        # Predicted value fields for highlighting
        self.predicted_value_field = QLineEdit()
        self.predicted_value_field.setReadOnly(True)
        self.predicted_value_field.setStyleSheet("background-color: #FFFAA0; color: #333333; font-weight: bold;")

        form_layout.addRow("Excitation Wavelength (nm):", self.exc_wavelength_input)
        form_layout.addRow("Excitation Power (W/cm²):", self.power_input)
        form_layout.addRow("Epsilon (M^-1 cm^-1):", self.epsilon_input)
        form_layout.addRow("Frame Length (s):", self.frame_length_input)
        form_layout.addRow("Avogadro's Number:", self.N_input)
        form_layout.addRow("Planck's Constant (J·s):", self.h_input)
        form_layout.addRow("Speed of Light (m/s):", self.c_input)

        # Estimated sigma noise input
        self.sigma_noise_input = QLineEdit("1.0")
        form_layout.addRow("Estimated Sigma Noise:", self.sigma_noise_input)

        form_layout.addRow("Predicted Value:", self.predicted_value_field)

        # Predict button
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self.predict_parameter)
        form_layout.addWidget(predict_button)

        # Add form layout to the main layout
        main_layout.addLayout(form_layout)

        # Create a Matplotlib figure and canvas to embed in the PySide6 application
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

        # Connect param_selector to dynamically update field visibility
        self.param_selector.currentIndexChanged.connect(self.update_fields)
        self.update_fields()

    def update_fields(self):
        selected_param = self.param_selector.currentText()

        if selected_param == "Epsilon":
            self.power_input.setEnabled(True)
            self.exc_wavelength_input.setEnabled(True)
            self.epsilon_input.setEnabled(False)
            self.epsilon_input.setPlaceholderText("Predicted value")

        elif selected_param == "Excitation Power":
            self.epsilon_input.setEnabled(True)
            self.exc_wavelength_input.setEnabled(True)
            self.power_input.setEnabled(False)
            self.power_input.setPlaceholderText("Predicted value")

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open .npy File", "", "Numpy Files (*.npy)")
        if file_path:
            self.photon_signal = np.load(file_path) + np.random.normal(scale=1)  # TODO: remove noise
            self.upload_button.setText(f"Loaded: {file_path.split('/')[-1]}")
            self.plot_photons()

    def predict_parameter(self):
        if self.photon_signal is None:
            print("No photon signal uploaded!")
            return

        self.predicted_value_field.clear()

        try:
            frame_length = float(self.frame_length_input.text())
            lambda_01 = np.mean(self.photon_signal) / frame_length  # Mean photon count / frame length
            exc_wavelength = float(self.exc_wavelength_input.text())
            N = float(self.N_input.text())
            h = float(self.h_input.text())
            c = float(self.c_input.text())
        except ValueError:
            print("Please enter valid numbers for the required fields.")
            return

        ln_10 = np.log(10)
        selected_param = self.param_selector.currentText()

        if selected_param == "Epsilon":
            try:
                power = float(self.power_input.text())
                epsilon = lambda_01 / (power * exc_wavelength * (1e-6 * ln_10 / (N * h * c)))
                self.epsilon_input.setText(f"{epsilon:.2e}")
                self.predicted_value_field.setText(f"Predicted Epsilon: {epsilon}")
            except ValueError:
                print("Please enter a valid number for Excitation Power.")

        elif selected_param == "Excitation Power":
            try:
                epsilon = float(self.epsilon_input.text())
                power = lambda_01 / (epsilon * exc_wavelength * (1e-6 * ln_10 / (N * h * c)))
                self.power_input.setText(f"{power:.2e}")
                self.predicted_value_field.setText(f"Predicted Power: {power}")
            except ValueError:
                print("Please enter a valid number for Epsilon.")

    def plot_photons(self):
        if self.photon_signal is None:
            print("No photon signal to plot!")
            return

        # Clear the figure to reset it
        self.figure.clear()

        # Create a GridSpec layout for three side-by-side plots
        gs = self.figure.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        # First plot: Original photon signal
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.set_facecolor('black')
        ax1.plot(self.photon_signal, color='#FFCCE1')  # Use the same color as before
        ax1.set_xlabel('Frame index', color='white')
        ax1.set_ylabel('Number of photons', color='white')
        ax1.set_title('Photon Emissions (Noisy)', color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Apply denoising
        self.denoise_signal()

        # Second plot: Denoised photon signal
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.set_facecolor('black')
        ax2.plot(self.denoised_signal, color='#FFCCE1')
        ax2.set_xlabel('Frame index', color='white')
        ax2.set_ylabel('Number of photons', color='white')
        ax2.set_title('Denoised Photon Emissions', color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Third plot: Histogram of Denoised Photon Counts
        ax3 = self.figure.add_subplot(gs[0, 2])
        ax3.set_facecolor('black')
        ax3.set_xlabel('Number of photons', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.set_title('Histogram of Photons (Denoised)', color='white')

        # Create histogram for denoised signal
        photon_nonzero = self.denoised_signal[self.denoised_signal > 0]
        counts, bins, patches = ax3.hist(photon_nonzero, bins=30, color='#FFA07A', edgecolor='white', density=True, alpha=0.6)

        # Compute the mean of the denoised signal
        mean_photon_count = np.mean(photon_nonzero)

        # Create a Poisson distribution with the calculated mean
        from scipy.stats import poisson
        x = np.arange(0, bins[-1] + 1)
        poisson_pmf = poisson.pmf(x, mean_photon_count)

        # Scale the Poisson PMF to match the histogram's density scaling
        ax3.plot(x, poisson_pmf, 'r-', lw=2, label=f'Poisson (mean={mean_photon_count:.2f})')

        # Add labels, grid, and legend
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, color='gray', linestyle='--', linewidth=0.5)
        ax3.legend(loc='upper right', fontsize=10, facecolor='black', edgecolor='white', labelcolor='white')


        # Refresh the canvas to show the plots
        self.canvas.draw()

    

    def denoise_signal(self):
        sigma = float(self.sigma_noise_input.text())  # Get sigma from user input
        self.denoised_signal = self.photon_signal - np.random.normal(scale=sigma)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create the ParametersPredictorPage window
    window = ParametersPredictorPage()
    window.show()

    # Start the PySide6 application event loop
    sys.exit(app.exec())
