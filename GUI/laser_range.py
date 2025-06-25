import sys
import os

sys.path.append("Fluorescence_Fluctuations_Simulation/")
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')
import numpy as np
from PySide6.QtWidgets import QFileDialog, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QGroupBox, QComboBox, QLabel,QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

import matplotlib.pyplot as plt
from scipy.stats import poisson
from joblib import Parallel, delayed

from simulate_ctmc import simulate_protocol


 # Define a helper function to simulate for a given power
def simulate_for_power(power, experiment_setup, molecule):
    # Update the excitation_P for the current power simulation
    experiment_setup["excitation_P"] = power
    
    # Run the simulation with the updated laser power
    _, _, _, photons = simulate_protocol(experiment_setup, molecule)
    
    # Calculate the mean number of photons
    mean_photons = np.mean(photons)
    return power, mean_photons



class LaserRangePage(QWidget):
    def __init__(self, stacked_widget=None):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.photons = None  # To store the photon data
        self.mean_photons_list = []  # To store the mean number of photons for each laser power
        self.valid_powers = []  # To store valid powers

        # Default values for the experiment setup dictionary
        self.experiment_setup = {
            "protocol": "FF",
            "experiment_duration": None,  # s
            "num_frames": 100,           # frames
            "frame_length": 1e-2,         # s
            "activation_rate" : None, 
            "excitation_wavelength": 647  # nm
        }

        self.molecule = {
            "epsilon": 239000,                          # Extinction coefficient M^-1 cm^-1
            "excitation_lifetime": 1e-9,                # Excited state lifetime, s (1 ns)
            "num_cycles_before_bleaching": 1e5,         # Number of cycles before bleaching
            "alpha_nr": 1e-8,                           # Proportional to Quantum yield
            "d_E": 0.5,                                 # Energy difference (eV)
            "alpha_isc": 1e7                            # Intersystem crossing rate
        }

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Use QGridLayout for the form fields in two columns
        form_layout = QGridLayout()

        # Column 1: Experiment Setup inputs
        form_layout.addWidget(QLabel("Protocol:"), 0, 0)
        self.protocol_input = QComboBox()
        self.protocol_input.addItems(["FF","Blinking",  "STORM", "PALM"])
        form_layout.addWidget(self.protocol_input, 0, 1)

        form_layout.addWidget(QLabel("Number of Frames:"), 1, 0)
        self.num_frames_input = QLineEdit(str(self.experiment_setup["num_frames"]))
        form_layout.addWidget(self.num_frames_input, 1, 1)

        form_layout.addWidget(QLabel("Frame Length (s):"), 2, 0)
        self.frame_length_input = QLineEdit(str(self.experiment_setup["frame_length"]))
        form_layout.addWidget(self.frame_length_input, 2, 1)

        form_layout.addWidget(QLabel("Excitation Wavelength (nm):"), 3, 0)
        self.excitation_wavelength_input = QLineEdit(str(self.experiment_setup["excitation_wavelength"]))
        form_layout.addWidget(self.excitation_wavelength_input, 3, 1)

        form_layout.addWidget(QLabel("Molecule Epsilon (M^-1 cm^-1):"), 4, 0)
        self.epsilon_input = QLineEdit(str(self.molecule["epsilon"]))
        form_layout.addWidget(self.epsilon_input, 4, 1)

        form_layout.addWidget(QLabel("Excitation Lifetime (s):"), 5, 0)
        self.exc_lifetime_input = QLineEdit(str(self.molecule["excitation_lifetime"]))
        form_layout.addWidget(self.exc_lifetime_input, 5, 1)

        form_layout.addWidget(QLabel("Number of Cycles Before Bleaching:"), 6, 0)
        self.cycles_before_bleaching_input = QLineEdit(str(self.molecule["num_cycles_before_bleaching"]))
        form_layout.addWidget(self.cycles_before_bleaching_input, 6, 1)

        form_layout.addWidget(QLabel("Alpha NR:"), 7, 0)
        self.alpha_nr_input = QLineEdit(str(self.molecule["alpha_nr"]))
        form_layout.addWidget(self.alpha_nr_input, 7, 1)

        form_layout.addWidget(QLabel("Energy Difference (eV):"), 8, 0)
        self.d_e_input = QLineEdit(str(self.molecule["d_E"]))
        form_layout.addWidget(self.d_e_input, 8, 1)

        form_layout.addWidget(QLabel("Alpha ISC:"), 9, 0)
        self.alpha_isc_input = QLineEdit(str(self.molecule["alpha_isc"]))
        form_layout.addWidget(self.alpha_isc_input, 9, 1)

        # Column 2: Laser Power Range and Criteria
        form_layout.addWidget(QLabel("Laser Power Range Start (W/cm²):"), 0, 2)
        self.excitation_P_start_input = QLineEdit("0.01")
        form_layout.addWidget(self.excitation_P_start_input, 0, 3)

        form_layout.addWidget(QLabel("Laser Power Range End (W/cm²):"), 1, 2)
        self.excitation_P_end_input = QLineEdit("10")
        form_layout.addWidget(self.excitation_P_end_input, 1, 3)

        form_layout.addWidget(QLabel("Number of Steps:"), 2, 2)
        self.excitation_P_steps_input = QLineEdit("50")
        form_layout.addWidget(self.excitation_P_steps_input, 2, 3)

        form_layout.addWidget(QLabel("Min Criteria Threshold (mean photons):"), 4, 2)
        self.min_criteria_threshold_input = QLineEdit("5.0")  # Default value for min criteria threshold
        form_layout.addWidget(self.min_criteria_threshold_input, 4, 3)

        # Add the Max Criteria Threshold input
        form_layout.addWidget(QLabel("Max Criteria Threshold (mean photons):"), 5, 2)
        self.max_criteria_threshold_input = QLineEdit("100.0")  # Default value for max criteria threshold
        form_layout.addWidget(self.max_criteria_threshold_input, 5, 3)

        # Add grid layout to the main layout
        main_layout.addLayout(form_layout)

        # Add QLabel for valid power range
        self.valid_range_label = QLabel("Valid Laser Power Range: Not calculated yet")
        self.valid_range_label.setStyleSheet("color: #FFFAA0; font-weight: bold;")
        main_layout.addWidget(self.valid_range_label)

        # Add the Matplotlib canvas and toolbar
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # Add buttons for simulate and go back
        button_layout = QHBoxLayout()
        simulate_button = QPushButton("Simulate Range")
        simulate_button.clicked.connect(self.simulate_laser_range)
        button_layout.addWidget(simulate_button)

        back_button = QPushButton("Back to Main")
        back_button.clicked.connect(self.go_back_to_main)
        button_layout.addWidget(back_button)

        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def simulate_laser_range(self):
        try:
            # Get the input values for the thresholds and laser power range
            min_criteria_threshold = float(self.min_criteria_threshold_input.text())
            max_criteria_threshold = float(self.max_criteria_threshold_input.text())
            excitation_P_start = float(self.excitation_P_start_input.text())
            excitation_P_end = float(self.excitation_P_end_input.text())
            excitation_P_steps = int(self.excitation_P_steps_input.text())
        except ValueError:
            print("Please enter valid numbers for the range and criteria thresholds.")
            return

        # Generate laser powers to test
        laser_powers = np.linspace(excitation_P_start, excitation_P_end, excitation_P_steps)

       
        # Use Joblib to run the simulations in parallel
        results = Parallel(n_jobs=-1)(delayed(simulate_for_power)(power, self.experiment_setup, self.molecule) for power in laser_powers)

        # Unpack the results
        laser_powers, mean_photons_list = zip(*results)

        # Find the first exceedance of the min and max criteria
        valid_min_power = None
        valid_max_power = None

        for i, mean_photons in enumerate(mean_photons_list):
            if mean_photons >= min_criteria_threshold and valid_min_power is None:
                valid_min_power = laser_powers[i]
            if mean_photons >= max_criteria_threshold:
                valid_max_power = laser_powers[i]
                break  # Stop once we have both min and max exceedances

        # Display valid powers and plot results
        if valid_min_power is not None and valid_max_power is not None:
            valid_range = f"{valid_min_power:.2f} - {valid_max_power:.2f} W/cm²"
            self.valid_range_label.setText(f"Valid Laser Power Range: {valid_range}")
        else:
            self.valid_range_label.setText("No valid power range found.")

        # Plot the evolution of mean number of photons vs laser power
        self.plot_photons_vs_power(laser_powers, mean_photons_list, valid_min_power, valid_max_power)


    
    

    def plot_photons_vs_power(self, laser_powers, mean_photons_list, valid_min_power,valid_max_power):
        self.figure.clear()

        # Create a plot for laser power vs. mean photon count
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('black')
        ax.plot(laser_powers, mean_photons_list, color='#FFCCE1', label='Mean Photon Count')
        ax.set_xlabel('Laser Power (W/cm²)', color='white')
        ax.set_ylabel('Mean Number of Photons', color='white')
        ax.set_title('Photon Emission vs Laser Power', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Draw vertical lines for the valid power range
        if valid_min_power is not None and valid_max_power is not None:
            ax.axvline(valid_min_power, color='green', linestyle='--', label='Min Valid Power')
            ax.axvline(valid_max_power, color='red', linestyle='--', label='Max Valid Power')

        # Add legend
        ax.legend(loc='upper right', fontsize=8)

        # Refresh the canvas
        self.canvas.draw()

    def go_back_to_main(self):
        if self.stacked_widget:
            self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = LaserRangePage()
    window.show()
    sys.exit(app.exec())
