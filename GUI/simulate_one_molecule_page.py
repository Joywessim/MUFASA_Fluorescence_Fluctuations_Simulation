import sys
import os

sys.path.append("Fluorescence_Fluctuations_Simulation/")
# Ensure Matplotlib uses PySide6
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit,
    QPushButton, QGroupBox, QComboBox, QFileDialog, QCheckBox
)
from PySide6.QtCore import Qt

from scipy.stats import poisson
import numpy as np

from continuous_time.simulate_ctmc import simulate_protocol  # Ensure this module is accessible

class SimulateOneMoleculePage(QWidget):
    def __init__(self, stacked_widget=None):
        super().__init__()

        self.stacked_widget = stacked_widget

        # Default values for the experiment setup dictionary
        self.experiment_setup = {
            "protocol": "Blinking",
            "experiment_duration": None,  # s
            "num_frames": 1000,           # frames
            "frame_length": 1e-2,         # s
            "excitation_P": 1,            # W/cm^2
            "activation_rate": None,      # W/cm^2
            "excitation_wavelength": 647  # nm
        }

        # Default values for the molecule parameters dictionary
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
        main_layout = QVBoxLayout()  # Main vertical layout

        # Create input fields for experiment setup and molecule properties
        form_layout = QHBoxLayout()

        # Experiment Setup Group
        experiment_group_box = QGroupBox("Experiment Setup")
        experiment_form_layout = QFormLayout()
        self.protocol_input = QComboBox()
        self.protocol_input.addItems(["Blinking", "FF", "STORM", "PALM"])
        self.protocol_input.setCurrentText(self.experiment_setup["protocol"])
        self.num_frames_input = QLineEdit(str(self.experiment_setup["num_frames"]))
        self.frame_length_input = QLineEdit(str(self.experiment_setup["frame_length"]))
        self.excitation_p_input = QLineEdit(str(self.experiment_setup["excitation_P"]))
        self.activation_rate_input = QLineEdit(str(self.experiment_setup["activation_rate"]) if self.experiment_setup["activation_rate"] is not None else "")
        self.excitation_wavelength_input = QLineEdit(str(self.experiment_setup["excitation_wavelength"]))
        experiment_form_layout.addRow("Protocol:", self.protocol_input)
        experiment_form_layout.addRow("Number of Frames:", self.num_frames_input)
        experiment_form_layout.addRow("Frame Length (s):", self.frame_length_input)
        experiment_form_layout.addRow("Excitation Power (W/cm²):", self.excitation_p_input)
        experiment_form_layout.addRow("Activation Power (W/cm²):", self.activation_rate_input)
        experiment_form_layout.addRow("Excitation Wavelength (nm):", self.excitation_wavelength_input)
        experiment_group_box.setLayout(experiment_form_layout)

        # Molecule Properties Group
        molecule_group_box = QGroupBox("Molecule Properties")
        molecule_form_layout = QFormLayout()
        self.epsilon_input = QLineEdit(str(self.molecule["epsilon"]))
        self.exc_lifetime_input = QLineEdit(str(self.molecule["excitation_lifetime"]))
        self.cycles_before_bleaching_input = QLineEdit(str(f'{self.molecule["num_cycles_before_bleaching"]:.1e}'))
        self.alpha_nr_input = QLineEdit(str(self.molecule["alpha_nr"]))
        self.d_e_input = QLineEdit(str(self.molecule["d_E"]))
        self.alpha_isc_input = QLineEdit(str(f'{self.molecule["alpha_isc"]:.1e}'))
        molecule_form_layout.addRow("Epsilon (M^-1 cm^-1):", self.epsilon_input)
        molecule_form_layout.addRow("Excitation Lifetime (s):", self.exc_lifetime_input)
        molecule_form_layout.addRow("Cycles Before Bleaching:", self.cycles_before_bleaching_input)
        molecule_form_layout.addRow("Alpha NR:", self.alpha_nr_input)
        molecule_form_layout.addRow("Energy Difference (eV):", self.d_e_input)
        molecule_form_layout.addRow("Alpha ISC:", self.alpha_isc_input)
        molecule_group_box.setLayout(molecule_form_layout)

        # Add groups to the form layout
        form_layout.addWidget(experiment_group_box)
        form_layout.addWidget(molecule_group_box)

        main_layout.addLayout(form_layout)

        # Create a Matplotlib figure and canvas to embed in the PySide6 application
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)  # Add the navigation toolbar

        # Apply custom styles to the toolbar to match the dark theme
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: #1a1a1a;  /* Dark background */
                border: none;
            }
            QToolButton {
                background: #333333;  /* Button background */
                color: #f0f0f0;       /* Button text/icon color */
                border: 1px solid #444444;
                border-radius: 3px;
                margin: 2px;
                padding: 5px;
            }
            QToolButton:pressed {
                background: #444444;  /* Slightly lighter when pressed */
            }
            QToolButton:hover {
                background: #4a4a4a;  /* Change color on hover */
            }
        """)

        # Add toolbar and canvas to the main layout
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas)

        # Create layout for fit checkboxes
        fit_checkbox_layout = QHBoxLayout()

        # Create checkboxes
        self.poisson_fit_checkbox = QCheckBox("Show Poisson Fit")
        self.poisson_fit_checkbox.setChecked(True)
        self.poisson_fit_checkbox.toggled.connect(self.toggle_poisson_fit)

        self.theoretical_fit_checkbox = QCheckBox("Show Theoretical Poisson Fit")
        self.theoretical_fit_checkbox.setChecked(True)
        self.theoretical_fit_checkbox.toggled.connect(self.toggle_theoretical_fit)

        # Add checkboxes to the layout
        fit_checkbox_layout.addWidget(self.poisson_fit_checkbox)
        fit_checkbox_layout.addWidget(self.theoretical_fit_checkbox)

        # Add the fit checkbox layout to the main layout
        main_layout.addLayout(fit_checkbox_layout)

        # Create a horizontal layout for the Simulate and Back to Main buttons
        button_layout = QHBoxLayout()

        # Create the Simulate button
        simulate_button = QPushButton("Simulate")
        simulate_button.clicked.connect(self.run_simulation)

        # Create the Back to Main button
        back_button = QPushButton("Back to Main")
        back_button.clicked.connect(self.go_back_to_main)

        # Create the Save Photons Emission button
        save_button = QPushButton("Save Photons Emission")
        save_button.clicked.connect(self.save_photons_emission)

        # Add buttons to the horizontal layout
        button_layout.addWidget(simulate_button)
        button_layout.addWidget(back_button)
        button_layout.addWidget(save_button)  # Add Save button

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("Fluorescence Fluctuations Simulation")
        self.setGeometry(50, 50, 1200, 800)  # Larger window size

        # Initially plot an empty plot
        self.plot_empty()

    def plot_empty(self):
        # Clear the figure to reset it
        self.figure.clear()

        # Create a GridSpec layout for three side-by-side plots
        gs = self.figure.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

        # First plot: Empty Molecule Dynamics
        ax1 = self.figure.add_subplot(gs[0, 0])
        ax1.set_facecolor('black')
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('State', color='white')
        ax1.set_title('Molecule Dynamics', color='white')
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['S0', 'S1', 'B'], color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Second plot: Empty Number of Photons per Frame
        ax2 = self.figure.add_subplot(gs[0, 1])
        ax2.set_facecolor('black')
        ax2.set_xlabel('Frame index', color='white')
        ax2.set_ylabel('Number of photons', color='white')
        ax2.set_title('Number of Photons per Frame', color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Third plot: Empty Histogram of Photons
        ax3 = self.figure.add_subplot(gs[0, 2])
        ax3.set_facecolor('black')
        ax3.set_xlabel('Number of photons', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.set_title('Histogram of Photons before bleaching', color='white')
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Refresh the canvas to show the empty plots
        self.canvas.draw()

    def run_simulation(self):
        
            # Update the experiment setup and molecule dictionaries with user inputs
            self.experiment_setup["protocol"] = self.protocol_input.currentText()
            self.experiment_setup["num_frames"] = int(self.num_frames_input.text())
            self.experiment_setup["frame_length"] = float(self.frame_length_input.text())
            self.experiment_setup["excitation_P"] = float(self.excitation_p_input.text())
            self.experiment_setup["activation_rate"] = float(self.activation_rate_input.text()) if self.activation_rate_input.text() else None
            self.experiment_setup["excitation_wavelength"] = int(self.excitation_wavelength_input.text())

            self.molecule["epsilon"] = float(self.epsilon_input.text())
            self.molecule["excitation_lifetime"] = float(self.exc_lifetime_input.text())
            self.molecule["num_cycles_before_bleaching"] = float(self.cycles_before_bleaching_input.text())
            self.molecule["alpha_nr"] = float(self.alpha_nr_input.text())
            self.molecule["d_E"] = float(self.d_e_input.text())
            self.molecule["alpha_isc"] = float(self.alpha_isc_input.text())

            # Simulate the data
            times, dynamics, Q, photons = simulate_protocol(self.experiment_setup, self.molecule)

            # Clear the figure and plot the new data
            self.figure.clear()

            # Create a GridSpec layout for three side-by-side plots
            gs = self.figure.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

            # First plot: Molecule Dynamics
            ax1 = self.figure.add_subplot(gs[0, 0])
            ax1.set_facecolor('black')
            ax1.step(times, dynamics, where='post', color='#CCF1FF')
            ax1.set_xlabel('Time (s)', color='white')
            ax1.set_ylabel('State', color='white')
            ax1.set_title(f'Molecule Dynamics for {self.experiment_setup["protocol"]}', color='white')

            if self.experiment_setup["protocol"] in ["Blinking", "STORM"]:
                ax1.set_yticks([0, 1, 2, 3])
                ax1.set_yticklabels(['S0', 'S1', 'T1', 'B'], color='white')
            elif self.experiment_setup["protocol"] == "PALM":
                ax1.set_yticks([-1, 0, 1, 2])
                ax1.set_yticklabels(['NA', 'S0', 'S1', 'B'], color='white')
            elif self.experiment_setup["protocol"] == "FF":
                ax1.set_yticks([0, 1, 2])
                ax1.set_yticklabels(['S0', 'S1', 'B'], color='white')

            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='white')
            ax1.grid(True, color='gray', linestyle='--', linewidth=0.5)

            # Second plot: Number of Photons per Frame
            ax2 = self.figure.add_subplot(gs[0, 1])
            ax2.set_facecolor('black')
            ax2.plot(photons, color='#FFCCE1')
            ax2.set_xlabel('Frame index', color='white')
            ax2.set_ylabel('Number of photons', color='white')
            ax2.set_title('Number of Photons per Frame', color='white')
            ax2.tick_params(axis='x', colors='white')
            ax2.tick_params(axis='y', colors='white')
            ax2.grid(True, color='gray', linestyle='--', linewidth=0.5)

            # Third plot: Histogram of Photons
            ax3 = self.figure.add_subplot(gs[0, 2])
            ax3.set_facecolor('black')

            # Filter out zeros from photons
            photons_nonzero = photons[photons != 0]

            # Plot the histogram
            counts, bins, patches = ax3.hist(
                photons_nonzero, bins=20, color='#FFA07A',
                edgecolor='white', density=True, alpha=0.6
            )

            # Fit and plot the Poisson distribution
            mean_photons = np.mean(photons_nonzero)
            x = np.arange(0, bins[-1] + 1)
            poisson_fit_values = poisson.pmf(x, mean_photons)
            poisson_fit_line, = ax3.plot(
                x, poisson_fit_values, 'r-', lw=1,
                label=f'Poisson fit (mean={mean_photons:.2f})'
            )

            # Fit and plot the Theoretical Poisson distribution
            if self.experiment_setup["protocol"] == "PALM":
                theoretical_mean = -self.experiment_setup["frame_length"] * Q[1, 1]
            else:
                theoretical_mean = -self.experiment_setup["frame_length"] * Q[0, 0]
            # Ensure the theoretical_mean is non-negative for Poisson PMF
            theoretical_mean = max(theoretical_mean, 0)
            poisson_theory_fit_values = poisson.pmf(x, theoretical_mean)
            poisson_theory_fit_line, = ax3.plot(
                x, poisson_theory_fit_values, 'y-', lw=1,
                label=f'Theoretical Poisson fit (mean={theoretical_mean:.2f})'
            )

            # Set visibility based on checkbox states
            poisson_fit_line.set_visible(self.poisson_fit_checkbox.isChecked())
            poisson_theory_fit_line.set_visible(self.theoretical_fit_checkbox.isChecked())

            # Store references to the fit lines
            self.poisson_fit_line = poisson_fit_line
            self.poisson_theory_fit_line = poisson_theory_fit_line

            # Labels and title
            ax3.set_xlabel('Number of photons', color='white')
            ax3.set_ylabel('Frequency', color='white')
            ax3.set_title('Histogram of Photons before bleaching', color='white')
            ax3.tick_params(axis='x', colors='white')
            ax3.tick_params(axis='y', colors='white')
            ax3.grid(True, color='gray', linestyle='--', linewidth=0.5)

            # Add a legend
            ax3.legend(
                loc='upper right', fontsize=5,
                facecolor='black', edgecolor='white', labelcolor='white'
            )

            # Refresh the canvas to show the plots
            self.canvas.draw()
            self.photons = photons  # Store the photons array for saving

    def toggle_poisson_fit(self, checked):
        """
        Slot to toggle the visibility of the Poisson fit line.
        """
        if hasattr(self, 'poisson_fit_line'):
            self.poisson_fit_line.set_visible(checked)
            self.canvas.draw()

    def toggle_theoretical_fit(self, checked):
        """
        Slot to toggle the visibility of the Theoretical Poisson fit line.
        """
        if hasattr(self, 'poisson_theory_fit_line'):
            self.poisson_theory_fit_line.set_visible(checked)
            self.canvas.draw()

    def go_back_to_main(self):
        if self.stacked_widget:
            self.stacked_widget.setCurrentIndex(0)  # Assuming the main page is at index 0

    def save_photons_emission(self):
        if hasattr(self, 'photons'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Photons Emission", "", "Numpy Files (*.npy)"
            )
            if file_path:
                np.save(file_path, self.photons)
                print(f"Photons emission saved to {file_path}")
        else:
            print("No photon emission data available. Please run the simulation first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply a dark theme to the PySide6 application
    dark_stylesheet = """
    QWidget {
        background-color: #2e2e2e;
        color: #f0f0f0;
    }
    QGroupBox {
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 5px;
    }
    QLineEdit, QComboBox, QPushButton, QCheckBox {
        background-color: #4a4a4a;
        border: 1px solid #666666;
        padding: 5px;
        border-radius: 3px;
        color: #f0f0f0;
    }
    QPushButton, QCheckBox {
        background-color: #565656;
        border: 1px solid #888888;
        border-radius: 5px;
        padding: 5px 10px;
        color: #f0f0f0;
    }
    QPushButton:hover, QCheckBox:hover {
        background-color: #6d6d6d;
    }
    """

    app.setStyleSheet(dark_stylesheet)

    window = SimulateOneMoleculePage()
    window.show()
    sys.exit(app.exec())
