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
from PySide6.QtWidgets import QScrollArea, QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QFileDialog, QCheckBox
from PySide6.QtWidgets import QGridLayout, QScrollArea, QWidget, QVBoxLayout, QCheckBox, QPushButton
from matplotlib.colors import LinearSegmentedColormap

import tifffile


from src.camera import apply_camera
from src.utils import colors_poisson_output, colors_us_output, color_mean_poisson,color_mean_poisson_pixel ,color_mean_us  ,color_mean_us_pixel  


class CameraSimulationResultsWindow(QWidget):
    def __init__(self, frames_us, frames_poisson, original_frames):
        super().__init__()
        self.frames_us = frames_us
        self.frames_poisson = frames_poisson
        self.original_frames = original_frames
        self.current_frame_index = 0  # Start with the first frame
        self.selected_pixel = None  # To store the coordinates of the clicked pixel
        self.pixel_selection_enabled = False
        self.display_noisy_mean = False  # Add this line to track the checkbox state.
        self.display_noisy_mean_in_the_pixel = False  # Add this line to track the checkbox state.
        self.display_mean_us = False  # Add this line to track the checkbox state.
        self.display_mean_us_in_the_pixel = False  # Add this line to track the checkbox state.



        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Camera Simulation Results")
        self.setGeometry(100, 100, 1200, 900)  # Slightly larger window

        layout = QVBoxLayout()

        # Create a Matplotlib figure and canvas to embed in the PySide6 application
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
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
        layout.addWidget(self.toolbar)

        # Create a new widget for the checkboxes and wrap it in a QScrollArea
        checkbox_widget = QWidget()
        checkbox_layout = QVBoxLayout(checkbox_widget)

        # Add checkboxes for mean display options
        self.noisy_mean_checkbox = QCheckBox("Display Mean of the Noisy Poisson output per frame")
        self.noisy_mean_checkbox.stateChanged.connect(self.toggle_noisy_mean_display)
        checkbox_layout.addWidget(self.noisy_mean_checkbox)

        self.noisy_mean_in_pixel_checkbox = QCheckBox("Display Mean of Noisy Poisson in selected pixel")
        self.noisy_mean_in_pixel_checkbox.stateChanged.connect(self.toggle_noisy_mean_in_pixel_display)
        checkbox_layout.addWidget(self.noisy_mean_in_pixel_checkbox)

        self.mean_us_checkbox = QCheckBox("Display Mean of US frames per frame")
        self.mean_us_checkbox.stateChanged.connect(self.toggle_mean_us_display)
        checkbox_layout.addWidget(self.mean_us_checkbox)

        self.mean_us_in_pixel_checkbox = QCheckBox("Display Mean of US in selected pixel")
        self.mean_us_in_pixel_checkbox.stateChanged.connect(self.toggle_mean_us_in_pixel_display)
        checkbox_layout.addWidget(self.mean_us_in_pixel_checkbox)


        self.enalble_selection = QCheckBox("Enable pixel selection")
        self.enalble_selection.stateChanged.connect(self.toggle_pixel_selection)
        checkbox_layout.addWidget(self.enalble_selection)

        # Wrap the checkbox layout in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(checkbox_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedHeight(150)  # Adjust the height as needed

        # Add the scroll area to the main layout
        layout.addWidget(scroll_area)

        # Create a QGridLayout for the buttons in a 2x2 grid
        button_grid = QGridLayout()

        # Create navigation buttons
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.show_prev_frame)

        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.show_next_frame)

        self.evolution_button = QPushButton("Show Fluorophore Evolutions")
        self.evolution_button.clicked.connect(self.show_fluorophore_evolutions)

        self.export_button = QPushButton("Exporter frames_poisson en TIFF")
        self.export_button.clicked.connect(self.export_frames_poisson)

        # Add the buttons to the grid layout
        button_grid.addWidget(self.prev_button, 0, 0)
        button_grid.addWidget(self.next_button, 0, 1)
        button_grid.addWidget(self.evolution_button, 1, 0)
        button_grid.addWidget(self.export_button, 1, 1)

        # Add the grid layout to the main layout
        layout.addLayout(button_grid)

        # Plot the first frame for all outputs
        self.plot_results()

        # Connect the click event for the Poisson plot
        self.canvas.mpl_connect("button_press_event", self.on_click)

        self.setLayout(layout)

        
    def show_fluorophore_evolutions(self):
        print("shapes verification = ",self.frames_us.shape,  self.original_frames.shape[1] )
        if self.selected_pixel:
            self.evolutions_window = FluorophoreEvolutionsWindow(
                self.selected_pixel,
                self.original_frames,
                self.original_frames.shape[1] // self.frames_us.shape[1] ,
            )
            self.evolutions_window.show()


    def toggle_noisy_mean_display(self, state):
        self.display_noisy_mean = bool(state)
        self.plot_results()

    def toggle_noisy_mean_in_pixel_display(self, state):
        self.display_noisy_mean_in_the_pixel = bool(state)
        self.plot_results()

    def toggle_mean_us_display(self, state):
        self.display_mean_us = bool(state)
        self.plot_results()

    def toggle_mean_us_in_pixel_display(self, state):
        self.display_mean_us_in_the_pixel = bool(state)
        self.plot_results()


    def plot_results(self):
        self.figure.clear()

        # Create a GridSpec layout with two rows and two columns
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.5)

        # Top-left plot: frames_us
        ax1 = self.figure.add_subplot(gs[0, 0])
        custom_cmap_us = LinearSegmentedColormap.from_list("custom_cmap", colors_us_output)
        cax1 = ax1.imshow(self.frames_us[self.current_frame_index], cmap=custom_cmap_us, aspect='auto')
        ax1.set_title(f"US Frame {self.current_frame_index + 1}", color='white')
        ax1.axis('on')  # Show the axis for reference
        cbar1 = self.figure.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(colors='white')

        # Bottom-left plot: frames_poisson with click event
        ax2 = self.figure.add_subplot(gs[1, 0])
        custom_cmap_poisson = LinearSegmentedColormap.from_list("custom_cmap", colors_poisson_output)
        cax2 = ax2.imshow(self.frames_poisson[self.current_frame_index], cmap=custom_cmap_poisson, aspect='auto')
        ax2.set_title(f"Poisson Frame {self.current_frame_index + 1}", color='white')
        ax2.axis('on')
        cbar2 = self.figure.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(colors='white')

        # Highlight the selected pixel if it exists
        if self.selected_pixel:
            y, x = self.selected_pixel
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)

            rect2 = Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='red', facecolor='none')
            ax1.add_patch(rect2)

        # Right plot: Evolution of the selected pixel
        ax3 = self.figure.add_subplot(gs[:, 1])
        ax3.set_facecolor('black')

        if self.selected_pixel:
            y, x = self.selected_pixel
            selected_pixel_values = self.frames_poisson[:, y, x]
            
            # Check if the selected pixel has only zeros
            if np.any(selected_pixel_values > 0):
                ax3.plot(selected_pixel_values, label=f"Poisson Pixel ({x}, {y})", color= colors_poisson_output[-1])
                ax3.set_title(f'Evolution in Pixel ({x}, {y})', color='white')
                ax3.set_xlabel('Frame index', color='white')
                ax3.set_ylabel('Number of photons', color='white')
            else:
                # Display message if the pixel has no data
                ax3.text(0.5, 0.5, f"No evolution available for pixel ({x}, {y})", color='white', ha='center', va='center')
        else:
            ax3.text(0.5, 0.5, 'Click a pixel to see its evolution', color='white', ha='center', va='center')

        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.grid(True, color='gray', linestyle='--', linewidth=0.5)
        

        # Plot the noisy mean for all frames
        if self.display_noisy_mean:
            mean_noisy_per_frame = np.mean(self.frames_poisson, axis=(1, 2))  # Mean across height and width for each frame
            ax3.plot(mean_noisy_per_frame, label="Mean Noisy Poisson per frame", color= color_mean_poisson, linestyle="-." )
        
        # Plot the noisy mean in the selected pixel over time
        if self.display_noisy_mean_in_the_pixel and self.selected_pixel:
            mean_noisy_in_pixel = np.mean(self.frames_poisson[:, y, x])  # Mean of the selected pixel across time
            ax3.axhline(mean_noisy_in_pixel, label=f"Mean in Poisson Pixel ({x}, {y})", color = color_mean_poisson_pixel, linestyle='--')
        
        # Plot the mean of US frames for all frames
        if self.display_mean_us:
            mean_us_per_frame = np.mean(self.frames_us, axis=(1, 2))  # Mean across height and width for each frame
            ax3.plot(mean_us_per_frame, label="Mean US per frame",color= color_mean_us, linestyle="-." )
        
        # Plot the mean of US in the selected pixel over time
        if self.display_mean_us_in_the_pixel and self.selected_pixel:
            mean_us_in_pixel = np.mean(self.frames_us[:, y, x])  # Mean of the selected US pixel across time
            ax3.axhline(mean_us_in_pixel, label=f"Mean in US Pixel ({x}, {y})", color=color_mean_us_pixel, linestyle='--')
        
        ax3.legend()
        self.figure.tight_layout()
        self.canvas.draw()


    def on_click(self, event):
        if self.pixel_selection_enabled and event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            self.selected_pixel = (y, x)  # Store the clicked pixel coordinates
            self.plot_results()  # Update the plot with the highlighted pixel

    def show_prev_frame(self):
        if self.current_frame_index > 0:
            self.current_frame_index -= 1
            self.plot_results()

    def show_next_frame(self):
        if self.current_frame_index < self.frames_us.shape[0] - 1:
            self.current_frame_index += 1
            self.plot_results()

    def export_frames_poisson(self):
        # Open a dialog box to choose the save path
        save_path, _ = QFileDialog.getSaveFileName(self, "Save TIFF", "", "TIFF Files (*.tiff *.tif)")
        
        if save_path:
            try:
                # Save the frames_poisson sequence as a multi-page TIFF
                tifffile.imwrite(save_path, self.frames_poisson)
                np.save(save_path, self.frames_poisson)
                QMessageBox.information(self, "Export Successful", f"The frames_poisson sequence has been saved to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while saving: {e}")

    def toggle_pixel_selection(self, state):
        """Enable or disable pixel selection based on the checkbox."""
        self.pixel_selection_enabled = bool(state)


class FluorophoreEvolutionsWindow(QWidget):
    def __init__(self, selected_pixel, original_frames, us_factor):
        super().__init__()
        self.selected_pixel = selected_pixel
        self.original_frames = original_frames
        self.us_factor = us_factor

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Fluorophore Evolutions")
        self.setGeometry(100, 100, 1200, 800)

        layout = QVBoxLayout()

        # Create a Matplotlib figure and canvas to embed in the PySide6 application
        self.figure = plt.figure(facecolor='black')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add Matplotlib toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
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
        layout.addWidget(self.toolbar)

        # Plot the fluorophore evolutions
        self.plot_fluorophore_evolutions()

        self.setLayout(layout)

    def plot_fluorophore_evolutions(self):
        self.figure.clear()

        # Determine the original coordinates based on the selected pixel
        y, x = self.selected_pixel
        original_x = x * self.us_factor
        original_y = y * self.us_factor

        # Define the grid size based on the upsampling factor
        grid_size = self.us_factor  # us_factor is the number of pixels in each dimension in the original image contributing to one pixel in the undersampled image

        if grid_size <= 0:
            QMessageBox.critical(self, "Error", "Invalid grid size. Cannot display fluorophore evolutions.")
            return

        # Filter only the non-zero fluorophores
        non_zero_fluorophores = []
        for i in range(grid_size):
            for j in range(grid_size):
                fluorophore_index_x = original_x + i
                fluorophore_index_y = original_y + j
                if (
                    0 <= fluorophore_index_x < self.original_frames.shape[2]
                    and 0 <= fluorophore_index_y < self.original_frames.shape[1]
                    and np.any(self.original_frames[:, fluorophore_index_y, fluorophore_index_x] > 0)
                ):
                    non_zero_fluorophores.append((fluorophore_index_x, fluorophore_index_y))

        if not non_zero_fluorophores:
            QMessageBox.information(self, "No Fluorophores", "No fluorophores detected in the selected pixel.")
            return

        # Adjust the grid size based on the number of non-zero fluorophores
        new_grid_size = int(np.ceil(np.sqrt(len(non_zero_fluorophores))))
        gs = self.figure.add_gridspec(new_grid_size, new_grid_size, wspace=0.4, hspace=0.4)

        # Plot the evolution of non-zero fluorophores in the undersampled pixel
        for idx, (fluorophore_index_x, fluorophore_index_y) in enumerate(non_zero_fluorophores):
            ax = self.figure.add_subplot(gs[idx // new_grid_size, idx % new_grid_size])
            ax.plot(self.original_frames[:, fluorophore_index_y, fluorophore_index_x], color='white')
            ax.set_title(f'({fluorophore_index_x}, {fluorophore_index_y})', color='white')
            ax.tick_params(axis='x', colors='white', labelsize=6)
            ax.tick_params(axis='y', colors='white', labelsize=6)
            ax.set_facecolor('black')
            ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        self.figure.tight_layout()
        self.canvas.draw()



if __name__ == "__main__":

    frames = np.load("frames.npy")
    # Create the application
    app = QApplication(sys.argv)

    # Simulate camera processing
    frames_us, frames_poisson, frames_blur = apply_camera(
        frames, int(frames.shape[0]), frames.shape[1:], alpha_detection=0.5, us_factor=4
    )

    # Create and show the CameraSimulationResultsWindow
    window = CameraSimulationResultsWindow(frames_us, frames_poisson, frames)
    window.show()

    # Run the application's event loop
    sys.exit(app.exec())
