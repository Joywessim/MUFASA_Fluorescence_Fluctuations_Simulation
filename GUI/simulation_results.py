import sys
import os

sys.path.append("Fluorescence_Fluctuations_Simulation/")
# Ensure Matplotlib uses PySide6
os.environ["QT_API"] = "pyside6"

import matplotlib
matplotlib.use('QtAgg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import QCheckBox,QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                               QLineEdit, QLabel, QRadioButton, QButtonGroup, QStackedWidget)
from continuous_time.camera import apply_camera, apply_camera_advanced
from camera_results import CameraSimulationResultsWindow
from continuous_time.utils import colors_fluctuations_output
from matplotlib.colors import LinearSegmentedColormap

class SimulationResultsWindow(QWidget):
    def __init__(self, frames):
        super().__init__()
        self.frames = frames
        self.current_frame_index = 0
        self.num_fluorophores = frames.shape[0]
        self.pixel_indices = self.select_cluster_pixels()  # Select a cluster of pixels
        self.is_playing = False  # Track if animation is playing
        self.timer = QTimer(self)  # Create a QTimer
        self.timer.timeout.connect(self.show_next_frame)  # Connect timer to show_next_frame

        self.init_ui()
        self.plot_results()
    def init_ui(self):
        self.setWindowTitle("Simulation Results")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()  # Main vertical layout to hold everything

        # Create a Matplotlib figure and canvas to embed in the PySide6 application
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)

        # Layout for the plots and input fields
        plot_layout = QHBoxLayout()

        # Left layout for the canvas
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        left_layout.addWidget(self.toolbar)

        plot_layout.addLayout(left_layout, stretch=3)

        # Right layout for the evolution plots and input fields (inside the black area)
        right_layout = QVBoxLayout()

        # --- Evolution plots (handled by plot_results) go here ---

        # Add input fields for camera simulation parameters (place them in the black area)
        input_fields_layout = QVBoxLayout()
        
        # Spacer to push the fields down into the black area
        right_layout.addStretch()  

        # Now, add the input fields for camera simulation
        camera_label = QLabel("Camera Options")
        camera_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        input_fields_layout.addWidget(camera_label)

        # Add radio buttons to select between Basic and Advanced models
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Camera Model:")
        self.basic_model_radio = QRadioButton("Basic Model")
        self.advanced_model_radio = QRadioButton("Advanced Model")
        self.basic_model_radio.setChecked(True)  # Default selection

        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.basic_model_radio)
        model_selection_layout.addWidget(self.advanced_model_radio)

        input_fields_layout.addLayout(model_selection_layout)

        # Connect radio button signals to toggle between Basic and Advanced input fields
        self.basic_model_radio.toggled.connect(self.toggle_camera_model)

        # Add a stacked widget for showing different input fields for each model
        self.stacked_widget = QStackedWidget()

        # Basic Model Input Layout
        self.basic_model_layout = QWidget()
        basic_layout = QVBoxLayout()

        self.alpha_detection_input = QLineEdit()
        self.alpha_detection_input.setPlaceholderText("alpha_detection (default 1)")
        basic_layout.addWidget(QLabel("alpha_detection:"))
        basic_layout.addWidget(self.alpha_detection_input)
        
        
        self.sigma_noise = QLineEdit()
        self.sigma_noise.setPlaceholderText("sigma_noise (default 1e-2)")
        basic_layout.addWidget(QLabel("sigma_noise:"))
        basic_layout.addWidget(self.sigma_noise)

        self.kernel_size = QLineEdit()
        self.kernel_size.setPlaceholderText("Kernel Size (default 5)")
        basic_layout.addWidget(QLabel("kernel_size:"))
        basic_layout.addWidget(self.kernel_size)

        self.us_factor_input = QLineEdit()
        self.us_factor_input.setPlaceholderText("us_factor (default 4)")
        basic_layout.addWidget(QLabel("us_factor:"))
        basic_layout.addWidget(self.us_factor_input)
        self.basic_model_layout.setLayout(basic_layout)

        # Advanced Model Input Layout
        self.advanced_model_layout = QWidget()
        advanced_layout = QVBoxLayout()
        self.qe_input = QLineEdit()
        self.qe_input.setPlaceholderText("QE (default 0.9)")
        advanced_layout.addWidget(QLabel("QE:"))
        advanced_layout.addWidget(self.qe_input)
        

        self.c = QLineEdit()
        self.c.setPlaceholderText("c (default 0.002 )")
        advanced_layout.addWidget(QLabel("c:"))
        advanced_layout.addWidget(self.c)


        self.sigma_R_input = QLineEdit()
        self.sigma_R_input.setPlaceholderText("sigma_R (default 74.4)")
        advanced_layout.addWidget(QLabel("sigma_R:"))
        advanced_layout.addWidget(self.sigma_R_input)
        self.em_gain_input = QLineEdit()
        self.em_gain_input.setPlaceholderText("EM Gain (default 300)")
        advanced_layout.addWidget(QLabel("EM Gain:"))
        advanced_layout.addWidget(self.em_gain_input)

        self.e_adu_input = QLineEdit()
        self.e_adu_input.setPlaceholderText("e_adu (default 45)")
        advanced_layout.addWidget(QLabel("e_adu:"))
        advanced_layout.addWidget(self.e_adu_input)

        self.BL_input = QLineEdit()
        self.BL_input.setPlaceholderText("BL (default 100)")
        advanced_layout.addWidget(QLabel("BL:"))
        advanced_layout.addWidget(self.BL_input)

        self.kernel_size = QLineEdit()
        self.kernel_size.setPlaceholderText("Kernel Size (default 5)")
        advanced_layout.addWidget(QLabel("kernel_size:"))
        advanced_layout.addWidget(self.kernel_size)

        self.us_factor_input = QLineEdit()
        self.us_factor_input.setPlaceholderText("us_factor (default 4)")
        advanced_layout.addWidget(QLabel("us_factor:"))
        advanced_layout.addWidget(self.us_factor_input)


        

        self.advanced_model_layout.setLayout(advanced_layout)
        self.basic_model_radio.setStyleSheet("""
            QRadioButton {
                color: white;  /* Text color for the label */
                border: 1px solid #5a5a5a;  /* Border color to make it visible */
                padding: 5px;
                border-radius: 5px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #5a5a5a;  /* Visible border around the circle */
                background-color: #3c3c3c;  /* Dark background for unchecked */
                border-radius: 8px;
            }
            QRadioButton::indicator:checked {
                background-color: blue;  /* Blue color when checked */
                border: 2px solid white;  /* White border for visibility when checked */
            }
            """)

        self.advanced_model_radio.setStyleSheet("""
                        QRadioButton {
                            color: white;  /* Text color for the label */
                            border: 1px solid #5a5a5a;  /* Border color to make it visible */
                            padding: 5px;
                            border-radius: 5px;
                        }
                        QRadioButton::indicator {
                            width: 16px;
                            height: 16px;
                            border: 2px solid #5a5a5a;  /* Visible border around the circle */
                            background-color: #3c3c3c;  /* Dark background for unchecked */
                            border-radius: 8px;
                        }
                        QRadioButton::indicator:checked {
                            background-color: blue;  /* Blue color when checked */
                            border: 2px solid white;  /* White border for visibility when checked */
                        }
                        """)

        # Add both layouts to the stacked widget
        self.stacked_widget.addWidget(self.basic_model_layout)
        self.stacked_widget.addWidget(self.advanced_model_layout)

        # Add stacked widget to input_fields_layout
        input_fields_layout.addWidget(self.stacked_widget)

        # Add input for background value
        self.background_input = QLineEdit()
        self.background_input.setPlaceholderText("Enter background value")
        input_fields_layout.addWidget(QLabel("Background value:"))
        input_fields_layout.addWidget(self.background_input)


        # Add the "Show Camera Simulation Results" button
        self.camera_button = QPushButton("Show Camera Simulation Results")
        self.camera_button.clicked.connect(self.open_camera_simulation_results)
        input_fields_layout.addWidget(self.camera_button)

        # Add a "Save Frames" button
        self.save_frames_button = QPushButton("Save Frames")
        self.save_frames_button.clicked.connect(self.save_frames)
        input_fields_layout.addWidget(self.save_frames_button)

        # Now we put the input fields into the black area
        right_layout.addLayout(input_fields_layout)

        plot_layout.addLayout(right_layout, stretch=1)

        layout.addLayout(plot_layout)

        # Bottom part for navigation buttons (like Play, Previous, etc.)
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.show_prev_frame)
        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.show_next_frame)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)

        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("Enter frame number")
        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.go_to_frame)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(QLabel("Frame:"))
        nav_layout.addWidget(self.frame_input)
        nav_layout.addWidget(self.go_button)
        nav_layout.addWidget(self.play_button)

        layout.addLayout(nav_layout)

        self.setLayout(layout)

            
    def toggle_play(self):
        if self.is_playing:
            self.timer.stop()  # Stop the timer if it's playing
            self.play_button.setText("Play")  # Change button text back to "Play"
        else:
            self.timer.start(200)  # Start the timer with a 200ms interval
            self.play_button.setText("Pause")  # Change button text to "Pause"
        self.is_playing = not self.is_playing  # Toggle the play state

    def open_camera_simulation_results(self):

        # Call apply_camera with the user-defined parameters and background value
        frames_us,frames_noisy = self.simulate_camera_model()

        self.camera_window = CameraSimulationResultsWindow(frames_us=frames_us, frames_poisson=frames_noisy, original_frames=self.frames)
        self.camera_window.show()

    def select_cluster_pixels(self):
        # Find non-zero pixels (corresponding to fluorophores)
        non_zero_indices = np.argwhere(self.frames[self.current_frame_index] > 0)

        if len(non_zero_indices) < 4:
            return non_zero_indices  # Return what we have if fewer than 4 fluorophores

        # Find a cluster of 4 pixels close to each other
        selected_pixels = []
        for i in range(len(non_zero_indices)):
            distances = np.linalg.norm(non_zero_indices - non_zero_indices[i], axis=1)
            neighbors = non_zero_indices[distances < 10]  # Adjust the distance threshold as needed
            if len(neighbors) >= 4:
                selected_pixels = neighbors[:4]
                break

        return selected_pixels if len(selected_pixels) >= 4 else non_zero_indices[:4]
    
    def plot_results(self):
            self.figure.clear()

            # Create a GridSpec layout with three columns: one for imshow, four for evolution plots in 2x2 configuration
            gs = self.figure.add_gridspec(3, 3, width_ratios=[2, 1, 1], wspace=0.5, hspace=0.5)

            # Left plot: The current frame with imshow
            ax1 = self.figure.add_subplot(gs[:2, 0])
            custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_fluctuations_output)

            cax = ax1.imshow(self.frames[self.current_frame_index], cmap=custom_cmap, aspect='equal')
            ax1.set_title(f"Frame {self.current_frame_index + 1}/{len(self.frames)}", color='white')
            ax1.axis('on')  # Show the axis for reference

            # Add a colorbar to the imshow plot
            cbar = self.figure.colorbar(cax, ax=ax1, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(colors='white')

            # Highlight the selected pixels with a rectangle
            if len(self.pixel_indices) == 4:
                x_coords, y_coords = zip(*self.pixel_indices)
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                rect = Rectangle((min_y - 0.5, min_x - 0.5), max_y - min_y + 1, max_x - min_x + 1,
                                linewidth=1, edgecolor='r', facecolor='none')
                ax1.add_patch(rect)

                # Bottom-left plot: Zoomed-in imshow of the selected red box
                ax_zoomed = self.figure.add_subplot(gs[2, 0])  # Position it below the main imshow
                zoomed_region = self.frames[self.current_frame_index, min_x:max_x + 1, min_y:max_y + 1]
                zoomed_cax = ax_zoomed.imshow(zoomed_region, cmap=custom_cmap, aspect='equal')
                ax_zoomed.set_title("Zoomed-in Region", color='white')
                
                # Set axis labels for the zoomed-in image
                ax_zoomed.set_xlabel('Pixel X', color='white')
                ax_zoomed.set_ylabel('Pixel Y', color='white')
                ax_zoomed.tick_params(axis='x', colors='white')
                ax_zoomed.tick_params(axis='y', colors='white')

                # Add a colorbar to the zoomed-in region
                zoomed_cbar = self.figure.colorbar(zoomed_cax, ax=ax_zoomed, fraction=0.046, pad=0.04)
                zoomed_cbar.ax.tick_params(colors='white')  # Set color for the colorbar ticks

            # Plot the evolution of the four selected pixels in a 2x2 grid on the right
            for i in range(4):
                ax = self.figure.add_subplot(gs[i // 2, (i % 2) + 1])
                ax.set_facecolor('black')

                if len(self.pixel_indices) > i:
                    x, y = self.pixel_indices[i]
                    ax.plot(self.frames[:self.current_frame_index + 1, x, y], color=colors_fluctuations_output[1])
                    ax.set_title(f'Pixel Evolution ({x}, {y})', color='white')
                else:
                    ax.text(0.5, 0.5, 'No pixel available', color='white', ha='center', va='center')

                ax.set_xlabel('Frame index', color='white')
                ax.set_ylabel('Number of photons', color='white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

            self.figure.tight_layout()
            self.canvas.draw()


    def show_prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.pixel_indices = self.select_cluster_pixels()
            self.plot_results()

    def show_next_frame(self):
        if self.current_frame_index < len(self.frames) - 1:
            self.current_frame_index += 1
            self.pixel_indices = self.select_cluster_pixels()
            self.plot_results()

    def go_to_frame(self):
        try:
            frame_number = int(self.frame_input.text()) - 1
            if 0 <= frame_number < len(frames.shape):
                self.current_frame_index = frame_number
                self.pixel_indices = self.select_cluster_pixels()
                self.plot_results()
            else:
                print("Frame number out of range")
        except ValueError:
            print("Invalid frame number")

    def save_frames(self):
        # Open a file dialog to choose where to save the frames
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Frames", "", "Numpy Files (*.npy);;All Files (*)"
        )
        if file_path:
            try:
                np.save(file_path, self.frames)
                print(f"Frames saved successfully to {file_path}")
            except Exception as e:
                print(f"Failed to save frames: {e}")
        else:
            print("Save operation cancelled.")

    def toggle_camera_model(self):
        """ Toggle between basic and advanced camera model input fields. """
        if self.basic_model_radio.isChecked():
            self.stacked_widget.setCurrentIndex(0)  # Show basic model inputs
        else:
            self.stacked_widget.setCurrentIndex(1)  # Show advanced model inputs

    def simulate_camera_model(self):
        """ Simulate the camera model based on user inputs. """
        
        # Get background value
        background_value = float(self.background_input.text()) if self.background_input.text() else 0
        kernel_size = float(self.kernel_size.text()) if self.kernel_size.text() else 5.0
        us_factor = int(self.us_factor_input.text()) if self.us_factor_input.text() else 4
        
        # Check which model is selected (Basic or Advanced)
        if self.basic_model_radio.isChecked():
            # Basic model inputs
            alpha_detection = float(self.alpha_detection_input.text()) if self.alpha_detection_input.text() else 1.0
            sigma_noise = float(self.sigma_noise.text()) if self.sigma_noise.text() else 1e-2
            
            print(f"Simulating Basic Camera Model with alpha_detection={alpha_detection}, us_factor={us_factor}")
            
            # Call the basic camera model function (apply_camera)
            frames_us, frames_noisy = apply_camera(
                self.frames, 
                self.frames.shape[0], 
                self.frames.shape[1:], 
                alpha_detection=alpha_detection, 
                us_factor=us_factor, 
                sigma_noise=sigma_noise,  # default value for sigma_noise in basic model
                kernel_size=kernel_size,     # default kernel size in basic model
                background=background_value
            )
            
        else:
            # Advanced model inputs
            QE = float(self.qe_input.text()) if self.qe_input.text() else 0.9
            c = float(self.c.text()) if self.c.text() else 0.002
            sigma_R = float(self.sigma_R_input.text()) if self.sigma_R_input.text() else 74.4
            EM_gain = int(self.em_gain_input.text()) if self.em_gain_input.text() else 300
            e_adu = int(self.e_adu_input.text()) if self.e_adu_input.text() else 45
            BL = int(self.BL_input.text()) if self.BL_input.text() else 100
            
            print(f"Simulating Advanced Camera Model with QE={QE}, c={c}, sigma_R={sigma_R}, EM_gain={EM_gain}, e_adu={e_adu}")
            
            # Call the advanced camera model function (apply_camera_advanced)
            frames_us, frames_noisy = apply_camera_advanced(
                self.frames, 
                self.frames.shape[0], 
                self.frames.shape[1:], 
                QE=QE, 
                sigma_R=sigma_R, 
                c=c, 
                EM_gain=EM_gain, 
                e_adu=e_adu, 
                BL=BL,            # Baseline value for the advanced model
                us_factor=us_factor,       # You can make this user-defined if necessary
                kernel_size=kernel_size,     # Gaussian blur kernel size
                background=background_value
            )

        return frames_us, frames_noisy
        

if __name__ == "__main__":
    # Create the application
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

    # Load the frames from a .npy file (assuming 'frames.npy' exists)
    if not os.path.exists("frames.npy"):
        print("frames.npy file not found.")
        sys.exit(1)
    frames = np.load("frames.npy")[0:3]

    # Create and show the SimulationResultsWindow
    window = SimulationResultsWindow(frames)
    window.show()

    # Run the application's event loop
    sys.exit(app.exec())

