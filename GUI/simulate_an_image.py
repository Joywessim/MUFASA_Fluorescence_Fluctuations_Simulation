import sys
import os

sys.path.append("MUFASA_Fluorescence_Fluctuations_Simulation/")
# Ensure Matplotlib uses PySide6
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import  QLabel, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton, QGroupBox, QComboBox, QSizePolicy, QProgressBar
from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import QCoreApplication
from matplotlib.colors import LinearSegmentedColormap

from simulate_ctmc import simulate_protocol
from utils import fluctuations_to_images

from utils import colors_fluctuations_output 

from GUI.simulation_results import SimulationResultsWindow

class SimulateStructurePage(QWidget):
    def __init__(self, stacked_widget=None):
        super().__init__()

        self.stacked_widget = stacked_widget

        # Default values for the experiment setup dictionary
        self.experiment_setup = {
            "protocol": "FF",
            "experiment_duration": None,  # s
            "num_frames": 500,  # frames
            "frame_length": 1e-2,  # s
            "excitation_P": 1,  # W/cm^2
            "activation_rate": None,  # W/cm^2
            "excitation_wavelength": 647  # nm
        }

        # Default values for the molecule parameters dictionary
        self.molecule = {
            "epsilon": 239000,  # Extinction coefficient M^-1 cm^-1
            "excitation_lifetime": 1e-9,  # Excited state lifetime, s (1 ns)
            "num_cycles_before_bleaching": 1e5,  # Number of cycles before bleaching
            "alpha_nr": 1e-8,  # Proportional to Quantum yield
            "d_E": 0.5,  # Energy difference (eV)
            "alpha_isc": 1e7  # Intersystem crossing rate
        }
        self.grid = None  # Initialize the grid
        self.num_fluorophores = None  # Initialize the number of fluorophores

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
        self.cycles_before_bleaching_input = QLineEdit(f"{self.molecule['num_cycles_before_bleaching']:.1e}")
        self.alpha_nr_input = QLineEdit(str(self.molecule["alpha_nr"]))
        self.d_e_input = QLineEdit(str(self.molecule["d_E"]))
        self.alpha_isc_input = QLineEdit(f"{self.molecule['alpha_isc']:.1e}")
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

        # Create a layout for the image display and buttons
        bottom_layout = QHBoxLayout()

        # Left corner layout for the image
        image_layout = QVBoxLayout()
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)  # Add the navigation toolbar
        image_layout.addWidget(self.toolbar)
        image_layout.addWidget(self.canvas)
        bottom_layout.addLayout(image_layout, 2)  # Assign smaller space to the image

        # Layout for the buttons and display fields
        button_layout = QVBoxLayout()
        
        # Add a QLabel for the "Work in progress" message
        self.status_label = QLabel("")
        button_layout.addWidget(self.status_label)


        # Form layout for grid_size, num_fluorophores, and num_fluorophores_to_simulate
        form_display_layout = QFormLayout()

        # Fields to display grid_size and num_fluorophores
        self.grid_size_display = QLineEdit()
        self.grid_size_display.setReadOnly(True)



        self.num_fluorophores_display = QLineEdit()
        self.num_fluorophores_display.setReadOnly(True)

        # New field for num_fluorophores_to_simulate
        self.num_fluorophores_to_simulate_display = QLineEdit()
        self.num_fluorophores_to_simulate_display.setPlaceholderText("Enter number to simulate")

        self.grid_size_display = QLineEdit()
        
        
        self.grid_size_display.setReadOnly(True)

        self.num_fluorophores_display = QLineEdit()
        self.num_fluorophores_display.setReadOnly(True)

        # New fields for user to input x_min, x_max, y_min, y_max
        self.x_min_input = QLineEdit()
        self.x_min_input.setPlaceholderText("X Min ")
        self.x_max_input = QLineEdit()
        self.x_max_input.setPlaceholderText("X Max ")  # Adjusted for 1500 + 1023
        self.y_min_input = QLineEdit()
        self.y_min_input.setPlaceholderText("Y Min ")
        self.y_max_input = QLineEdit()
        self.y_max_input.setPlaceholderText("Y Max")  # Adjusted for 1500 + 1023

        # Add form rows for user inputs
        form_display_layout.addRow("X Min:", self.x_min_input)
        form_display_layout.addRow("X Max:", self.x_max_input)
        form_display_layout.addRow("Y Min:", self.y_min_input)
        form_display_layout.addRow("Y Max:", self.y_max_input)

        # Add form rows
        form_display_layout.addRow("Grid Size =", self.grid_size_display)
        form_display_layout.addRow("Number of Fluorophores =", self.num_fluorophores_display)
        form_display_layout.addRow("Number to Simulate =", self.num_fluorophores_to_simulate_display)

        # Add form layout to the button layout
        button_layout.addLayout(form_display_layout)

        # Create the Simulate button
        simulate_button = QPushButton("Simulate Structure")
        simulate_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Make button smaller
        simulate_button.clicked.connect(self.run_simulation)

        # Create the Import Positions Array button
        import_button = QPushButton("Import Positions Array")
        import_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Make button smaller
        import_button.clicked.connect(self.import_positions_array)


        # Create the Update Region button
        update_button = QPushButton("Update Region")
        update_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        update_button.clicked.connect(self.update_image_with_new_bounds)

        # Create the Back to Main button
        back_button = QPushButton("Back to Main")
        back_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Make button smaller
        back_button.clicked.connect(self.go_back_to_main)

        # Add buttons to the button layout
        button_layout.addWidget(import_button)
        button_layout.addWidget(update_button)
        button_layout.addWidget(simulate_button)
        button_layout.addWidget(back_button)
        


        button_layout.addStretch()  # Push buttons to the top

        bottom_layout.addLayout(button_layout, 3)  # Assign larger space to the buttons and fields
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("Fluorescence Fluctuations Simulation - Structure")
        self.setGeometry(50, 50, 1200, 800)  # Adjust window size as needed

    

    def go_back_to_main(self):
        if self.stacked_widget:
            self.stacked_widget.setCurrentIndex(0)  # Assuming the main page is at index 0
            
    def import_positions_array(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Positions CSV", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            # Load the CSV file and process it as described
            df = pd.read_csv(file_name, header=None)
            df.columns = ['x', 'y', 'z']

            # Get x and y from the DataFrame
            x, y = df['x'], df['y']

            # Get user-defined min and max values, or use defaults
            x_min = int(self.x_min_input.text()) if self.x_min_input.text() else 0
            x_max = int(self.x_max_input.text()) if self.x_max_input.text() else x.max()
            y_min = int(self.y_min_input.text()) if self.y_min_input.text() else 0
            y_max = int(self.y_max_input.text()) if self.y_max_input.text() else y.max()

            # Filter the positions based on user input
            filtered_positions = df[(x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)]

            # Get the filtered x and y positions
            filtered_x = filtered_positions['x']
            filtered_y = filtered_positions['y']

            # Normalize the coordinates to fit within the grid
            grid_size = max(int(x_max - x_min + 1), int(y_max - y_min + 1))  # Ensure the grid size covers the specified range
            grid = np.zeros((grid_size, grid_size), dtype=int)

            # Adjust the positions to the grid
            adjusted_x = (filtered_x - x_min).astype(int)
            adjusted_y = (filtered_y - y_min).astype(int)

            self.adjusted_x = adjusted_x
            self.adjusted_y = adjusted_y

            # Place the fluorophores in the grid
            for x_pos, y_pos in zip(adjusted_x, adjusted_y):
                grid[x_pos, y_pos] = 1

            # Store the grid and number of fluorophores for further use
            self.grid = grid
            self.num_fluorophores = filtered_x.shape[0]

            # Update the display fields with the grid size and number of fluorophores
            self.grid_size_display.setText(f"{self.grid.shape[0]} x {self.grid.shape[1]}")
            self.num_fluorophores_display.setText(str(self.num_fluorophores))

            # Display the image after importing
            self.display_image()


 
 
 


    def display_image(self):
        # Display the grid using imshow
        self.figure.clear()
        ax = self.figure.add_subplot(111)  # Add a subplot to handle the imshow

        # Create a masked grid to highlight non-zero values

        # Define a custom colormap
        # Define a custom colormap
        
        contours = ax.contour(self.grid, colors=colors_fluctuations_output[-1], linewidths=0.5)

        # Display the masked grid with the custom colormap
        cax = ax.imshow(self.grid, cmap="grey", vmin=0, vmax=1)  # Set limits to 0 and 1

        # Set the title for the plot
        ax.set_title("Fluorescence Image of Fluorophores", color='white')

        # Add a colorbar associated with the current Axes
        cbar = self.figure.colorbar(cax, ax=ax)
        cbar.ax.tick_params(colors='white')  # Set colorbar ticks color to black

        # Adjust layout to prevent overlap
        self.figure.tight_layout()

        self.canvas.draw()


    def run_simulation(self):


        

        # Read the number of fluorophores to simulate
        try:
            num_fluorophores_to_run = int(self.num_fluorophores_to_simulate_display.text())
            if num_fluorophores_to_run > self.num_fluorophores:
                print("Number to Simulate cannot be greater than the total number of fluorophores.")
                return
        except ValueError:
            print("Invalid input for Number to Simulate.")
            return

        # Prepare the experiment setup and molecule parameters
        experiment_setup = {
            "protocol": self.protocol_input.currentText(),
            "experiment_duration": self.experiment_setup["experiment_duration"],
            "num_frames": int(self.num_frames_input.text()),
            "frame_length": float(self.frame_length_input.text()),
            "excitation_P": float(self.excitation_p_input.text()),
            "activation_rate": float(self.activation_rate_input.text()) if self.activation_rate_input.text() else None,
            "excitation_wavelength": int(self.excitation_wavelength_input.text())
        }

        molecule = {
            "epsilon": float(self.epsilon_input.text()),
            "excitation_lifetime": float(self.exc_lifetime_input.text()),
            "num_cycles_before_bleaching": float(self.cycles_before_bleaching_input.text()),
            "alpha_nr": float(self.alpha_nr_input.text()),
            "d_E": float(self.d_e_input.text()),
            "alpha_isc": float(self.alpha_isc_input.text())
        }

        # Show "Work in progress" message when simulation starts
        self.status_label.setText("Simulating Molecules...")
        self.status_label.setStyleSheet("color: #CCF1FF;")

        # Ensure the UI updates the label text immediately
        QCoreApplication.processEvents()

        results = Parallel(n_jobs=-1)(
            delayed(simulate_protocol)(experiment_setup, molecule)
            for _ in tqdm(range(num_fluorophores_to_run))
        )
        self.status_label.setText("")

        # Show "Work in progress" message when simulation starts
        self.status_label.setText("Creating frames...")
        self.status_label.setStyleSheet("color: #CCF1FF;")

        # Ensure the UI updates the label text immediately
        QCoreApplication.processEvents()
        
        # Convert fluctuations (photon counts) to images
        photons_per_frame = np.zeros((num_fluorophores_to_run, experiment_setup['num_frames']))
        for i in range(len(results)):
            photons_per_frame[i, :] = results[i][-1]

        
        frames = fluctuations_to_images(photons_per_frame, experiment_setup['num_frames'], self.grid.shape, self.adjusted_x, self.adjusted_y)
        frames = np.array(frames)

        
        # Clear the "Work in progress" message when done
        self.status_label.setText("")


        # Open the new window to display the frames
        self.results_window = SimulationResultsWindow(frames)
        self.results_window.show()

    def update_image_with_new_bounds(self):
        # Get user-defined min and max values
        x_min = int(self.x_min_input.text()) if self.x_min_input.text() else 0
        x_max = int(self.x_max_input.text()) if self.x_max_input.text() else self.grid.shape[0]   # Adjust based on current grid
        y_min = int(self.y_min_input.text()) if self.y_min_input.text() else 0
        y_max = int(self.y_max_input.text()) if self.y_max_input.text() else self.grid.shape[1] # Adjust based on current grid

        # Filter the positions based on updated user input
        filtered_positions = pd.DataFrame({
            'x': self.adjusted_x + x_min,
            'y': self.adjusted_y + y_min
        })
        filtered_positions = filtered_positions[
            (filtered_positions['x'] >= x_min) & (filtered_positions['x'] <= x_max) &
            (filtered_positions['y'] >= y_min) & (filtered_positions['y'] <= y_max)
        ]

        # Get the filtered x and y positions
        filtered_x = filtered_positions['x']
        filtered_y = filtered_positions['y']

        # Normalize the coordinates to fit within the new grid
        new_grid_size = max(x_max - x_min + 1, y_max - y_min + 1)
        new_grid = np.zeros((new_grid_size, new_grid_size), dtype=int)

        # Adjust the positions to the new grid
        adjusted_x = (filtered_x - x_min).astype(int)
        adjusted_y = (filtered_y - y_min).astype(int)

        # Place the fluorophores in the new grid
        for x_pos, y_pos in zip(adjusted_x, adjusted_y):
            new_grid[x_pos, y_pos] = 1

        # Update the grid and number of fluorophores
        self.grid = new_grid
        self.num_fluorophores = filtered_x.shape[0]

        # Update the display fields with the new grid size and number of fluorophores
        self.grid_size_display.setText(f"{self.grid.shape[0]} x {self.grid.shape[1]}")
        self.num_fluorophores_display.setText(str(self.num_fluorophores))

        # Display the updated image
        self.display_image()

   

 



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
    QLineEdit, QComboBox, QPushButton {
        background-color: #4a4a4a;
        border: 1px solid #666666;
        padding: 5px;
        border-radius: 3px;
        color: #f0f0f0;
    }
    QPushButton {
        background-color: #565656;
        border: 1px solid #888888;
        border-radius: 5px;
        padding: 5px 10px;
        color: #f0f0f0;
    }
    QPushButton:hover {
        background-color: #6d6d6d;
    }
    """
    app.setStyleSheet(dark_stylesheet)

    window = SimulateStructurePage()
    window.show()
    sys.exit(app.exec())

