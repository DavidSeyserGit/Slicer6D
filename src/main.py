import sys
import os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QLabel, QSplitter, QStatusBar, QScrollArea
)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor
import tempfile

# Import our modules
import deformer
import slicing  # Assuming your slicer file is named slicing.py
import gcode_optimizer # Assuming your optimizer file is named gcode_optimizer.py


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Print Overhang Deformer")
        self.resize(1200, 800)

        # Model data
        self.original_mesh = None
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = None
        self.base_mesh = None # Store the pristine loaded mesh

        # Slice data for visualization
        self.slice_z_levels = None
        self.slice_layers = None
        self.slice_paths_actor_name = "slice_paths" # Name for PyVista actor

        # Central widget & layout
        central = QWidget()
        self.setCentralWidget(central)
        hl = QHBoxLayout(central)

        # Splitter: controls | viewer
        splitter = QSplitter(Qt.Horizontal)
        hl.addWidget(splitter)

        # --- Control panel setup with ScrollArea ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(300)
        scroll_area.setMaximumWidth(400)
        scroll_content = QWidget()
        vctrl = QVBoxLayout(scroll_content)

        self._build_file_group(vctrl)
        self._build_view_group(vctrl) # Modified
        self._build_orientation_group(vctrl)
        self._build_overhang_group(vctrl)
        self._build_deform_group(vctrl)
        self._build_slice_group(vctrl)

        vctrl.addStretch()
        scroll_area.setWidget(scroll_content)
        splitter.addWidget(scroll_area)
        # --- End of ScrollArea setup ---

        # --- Viewer
        viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(viewer_frame)
        self.plotter = QtInteractor(viewer_frame)
        viewer_layout.addWidget(self.plotter)
        splitter.addWidget(viewer_frame)

        splitter.setSizes([350, 850])

        # Initial plotter setup
        self.plotter.set_background([0.2, 0.2, 0.2])
        self.plotter.add_axes(interactive=True, line_width=3)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.plotter.add_text("Load a model to begin", font_size=16,
                              position="upper_left", name="status_text") # Added name

    # --- Build Groups (File, Orientation, Overhang, Deform, Slice - unchanged) ---
    def _build_file_group(self, parent):
        gb = QGroupBox("File")
        l = QVBoxLayout(gb)
        btn = QPushButton("Load STL")
        btn.clicked.connect(self.load_model)
        l.addWidget(btn)
        hb = QHBoxLayout()
        b1 = QPushButton("Export Orig")
        b1.clicked.connect(self.export_original)
        b2 = QPushButton("Export Def")
        b2.clicked.connect(self.export_deformed)
        hb.addWidget(b1); hb.addWidget(b2)
        l.addLayout(hb)
        parent.addWidget(gb)

    def _build_orientation_group(self, parent):
        gb = QGroupBox("Print Orientation")
        layout = QVBoxLayout(gb)
        h = QHBoxLayout()
        h.addWidget(QLabel("Base:"))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems([
            "Bottom (XY)", "Front (XZ)", "Side (YZ)",
            "Top (-XY)", "Back (-XZ)", "Other (-YZ)"
        ])
        h.addWidget(self.orient_combo)
        btn = QPushButton("Set Preset")
        btn.clicked.connect(self.set_orientation_preset) # Renamed for clarity
        h.addWidget(btn)
        layout.addLayout(h)
        hx = QHBoxLayout()
        hx.addWidget(QLabel("Rotate X:"))
        self.rot_x_sb = QDoubleSpinBox()
        self.rot_x_sb.setRange(-180, 180); self.rot_x_sb.setSingleStep(1); self.rot_x_sb.setSuffix("°")
        self.rot_x_sb.valueChanged.connect(self.apply_manual_rotation)
        hx.addWidget(self.rot_x_sb)
        layout.addLayout(hx)
        hy = QHBoxLayout()
        hy.addWidget(QLabel("Rotate Y:"))
        self.rot_y_sb = QDoubleSpinBox()
        self.rot_y_sb.setRange(-180, 180); self.rot_y_sb.setSingleStep(1); self.rot_y_sb.setSuffix("°")
        self.rot_y_sb.valueChanged.connect(self.apply_manual_rotation)
        hy.addWidget(self.rot_y_sb)
        layout.addLayout(hy)
        hz = QHBoxLayout()
        hz.addWidget(QLabel("Rotate Z:"))
        self.rot_z_sb = QDoubleSpinBox()
        self.rot_z_sb.setRange(-180, 180); self.rot_z_sb.setSingleStep(1); self.rot_z_sb.setSuffix("°")
        self.rot_z_sb.valueChanged.connect(self.apply_manual_rotation)
        hz.addWidget(self.rot_z_sb)
        layout.addLayout(hz)
        self.bed_cb = QCheckBox("Show Print Bed")
        self.bed_cb.setChecked(True)
        self.bed_cb.stateChanged.connect(self.update_display) # Use update_display
        layout.addWidget(self.bed_cb)
        parent.addWidget(gb)

    def _build_overhang_group(self, parent):
        gb = QGroupBox("Overhang Detection")
        l = QVBoxLayout(gb)
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Angle:"))
        self.ang_sb = QSpinBox(); self.ang_sb.setRange(20,90); self.ang_sb.setValue(45)
        self.ang_sb.setSuffix("°"); h1.addWidget(self.ang_sb)
        l.addLayout(h1)
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Support Dist:"))
        self.sup_sb = QDoubleSpinBox(); self.sup_sb.setRange(0.1,20.0)
        self.sup_sb.setValue(5.0); self.sup_sb.setSuffix("mm")
        self.sup_sb.setSingleStep(0.5); h2.addWidget(self.sup_sb)
        l.addLayout(h2)
        self.ray_cb = QCheckBox("Use Ray Trace"); self.ray_cb.setChecked(True)
        l.addWidget(self.ray_cb)
        btn = QPushButton("Detect Overhangs"); btn.clicked.connect(self.detect_overhangs)
        l.addWidget(btn)
        parent.addWidget(gb)

    def _build_deform_group(self, parent):
        gb = QGroupBox("Deformation")
        l = QVBoxLayout(gb)
        hb = QHBoxLayout()
        b1 = QPushButton("Deform"); b1.clicked.connect(self.deform_model)
        b2 = QPushButton("Reset"); b2.clicked.connect(self.reset_model)
        hb.addWidget(b1); hb.addWidget(b2)
        l.addLayout(hb)
        parent.addWidget(gb)

    def _build_slice_group(self, parent):
        """Create the Slicing controls panel."""
        gb = QGroupBox("Slicing")
        l = QVBoxLayout(gb)
        # layer height
        hl = QHBoxLayout()
        hl.addWidget(QLabel("Layer Height:"))
        self.slice_layer_sb = QDoubleSpinBox()
        self.slice_layer_sb.setRange(0.05, 5.0); self.slice_layer_sb.setSingleStep(0.05)
        self.slice_layer_sb.setValue(0.2); self.slice_layer_sb.setSuffix(" mm")
        hl.addWidget(self.slice_layer_sb)
        l.addLayout(hl)
        # extrusion
        he = QHBoxLayout()
        he.addWidget(QLabel("Extrusion:"))
        self.slice_extrude_sb = QDoubleSpinBox()
        self.slice_extrude_sb.setRange(0.01, 1.0); self.slice_extrude_sb.setSingleStep(0.01)
        self.slice_extrude_sb.setValue(0.05)
        he.addWidget(self.slice_extrude_sb)
        he.addWidget(QLabel("E/mm"))
        l.addLayout(he)
        # feedrate
        hf = QHBoxLayout()
        hf.addWidget(QLabel("Print Feed:"))
        self.slice_feed_sb = QSpinBox()
        self.slice_feed_sb.setRange(100, 5000); self.slice_feed_sb.setValue(1500)
        hf.addWidget(self.slice_feed_sb)
        l.addLayout(hf)
        # travel feed
        ht = QHBoxLayout()
        ht.addWidget(QLabel("Travel Feed:"))
        self.slice_travel_sb = QSpinBox()
        self.slice_travel_sb.setRange(100, 8000); self.slice_travel_sb.setValue(3000)
        ht.addWidget(self.slice_travel_sb)
        l.addLayout(ht)

        # --- Infill Controls ---
        h_inf_s = QHBoxLayout()
        h_inf_s.addWidget(QLabel("Infill Spacing:"))
        self.slice_infill_spacing_sb = QDoubleSpinBox()
        self.slice_infill_spacing_sb.setRange(0.1, 20.0) # Min > 0
        self.slice_infill_spacing_sb.setSingleStep(0.5)
        self.slice_infill_spacing_sb.setValue(2.0)
        self.slice_infill_spacing_sb.setSuffix(" mm")
        h_inf_s.addWidget(self.slice_infill_spacing_sb)
        l.addLayout(h_inf_s)

        h_inf_a = QHBoxLayout()
        h_inf_a.addWidget(QLabel("Infill Angle:"))
        self.slice_infill_angle_sb = QSpinBox() # SpinBox is fine for degrees
        self.slice_infill_angle_sb.setRange(0, 179)
        self.slice_infill_angle_sb.setValue(45)
        self.slice_infill_angle_sb.setSuffix(" °")
        h_inf_a.addWidget(self.slice_infill_angle_sb)
        l.addLayout(h_inf_a)
        # --- End Infill Controls ---

        # options
        self.slice_deformed_cb = QCheckBox("Use Deformed Mesh")
        l.addWidget(self.slice_deformed_cb)
        self.slice_optimize_cb = QCheckBox("Optimize G‑code")
        self.slice_optimize_cb.setChecked(True)
        l.addWidget(self.slice_optimize_cb)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        slice_btn = QPushButton("Slice to G‑code")
        slice_btn.setMinimumHeight(36)
        slice_btn.clicked.connect(self.slice_model)
        btn_layout.addWidget(slice_btn)
        vis_btn = QPushButton("Show Paths")
        vis_btn.setMinimumHeight(36)
        vis_btn.clicked.connect(self.display_slice_paths)
        btn_layout.addWidget(vis_btn)
        l.addLayout(btn_layout)
        parent.addWidget(gb)

    def _build_view_group(self, parent):
        """Build the View control group."""
        gb = QGroupBox("View")
        g = QGridLayout(gb) # Use grid for better layout

        # Camera preset buttons
        bxy = QPushButton("Top XY"); bxy.clicked.connect(lambda: self.plotter.view_xy())
        bxz = QPushButton("Front XZ"); bxz.clicked.connect(lambda: self.plotter.view_xz())
        byz = QPushButton("Side YZ"); byz.clicked.connect(lambda: self.plotter.view_yz())
        bis = QPushButton("Iso"); bis.clicked.connect(lambda: self.plotter.view_isometric())
        g.addWidget(bxy, 0, 0); g.addWidget(bxz, 0, 1)
        g.addWidget(byz, 1, 0); g.addWidget(bis, 1, 1)

        # Display option checkboxes
        self.mesh_vis_cb = QCheckBox("Show Main Mesh") # New checkbox
        self.mesh_vis_cb.setChecked(True)
        self.mesh_vis_cb.stateChanged.connect(self.update_display) # Connect to update
        g.addWidget(self.mesh_vis_cb, 2, 0, 1, 2) # Span across two columns

        self.wf_cb = QCheckBox("Wireframe"); self.wf_cb.stateChanged.connect(self.update_display)
        self.ed_cb = QCheckBox("Show Edges"); self.ed_cb.setChecked(True)
        self.ed_cb.stateChanged.connect(self.update_display)
        g.addWidget(self.wf_cb, 3, 0); g.addWidget(self.ed_cb, 3, 1)

        parent.addWidget(gb)


    def slice_model(self):
        """Slice the currently loaded mesh (original or deformed) with infill."""
        # Reset previous slice data first
        self.slice_z_levels = None
        self.slice_layer_polygons = None # Changed variable name
        self.generated_slice_paths = None # ADDED: To store paths from gcode generation
        self.plotter.remove_actor(self.slice_paths_actor_name) # Clear old paths from view

        mesh_to_slice = None
        if self.slice_deformed_cb.isChecked() and self.deformed_mesh is not None:
            mesh_to_slice = self.deformed_mesh
            source_name = "deformed"
        elif self.original_mesh is not None:
            mesh_to_slice = self.original_mesh
            source_name = "original"
        else:
            self.statusBar().showMessage("No mesh loaded to slice")
            return

        # Ask for base name for output files
        fn, _ = QFileDialog.getSaveFileName(
            self, "Save G‑code Base Name", "", "G‑code Files (*.gcode)")
        if not fn:
            return # User cancelled

        base_path, _ = os.path.splitext(fn)
        raw_gcode_path = base_path + "_raw.gcode"
        opt_gcode_path = base_path + "_opt.gcode"

        tmp_stl_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_stl_file = tmp.name
            mesh_to_slice.save(tmp_stl_file)
            self.statusBar().showMessage(f"Slicing {source_name} mesh...")
            QApplication.processEvents()

            # --- Get slicing parameters from GUI (including infill) ---
            layer_h = self.slice_layer_sb.value()
            ext_mm  = self.slice_extrude_sb.value()
            feed    = self.slice_feed_sb.value()
            travel  = self.slice_travel_sb.value()
            infill_space = self.slice_infill_spacing_sb.value() # Get infill spacing
            infill_angle = self.slice_infill_angle_sb.value()   # Get infill angle
            # --- End Get Parameters ---

            z_levels, layer_polygons = slicing.slice_mesh_to_polygons(tmp_stl_file, layer_h)
            self.slice_z_levels = z_levels         # Store Z levels
            self.slice_layer_polygons = layer_polygons # Store polygons

            if not z_levels or not layer_polygons or all(not lp for lp in layer_polygons):
                 self.statusBar().showMessage("Slicing completed, but no valid layers found.")
                 self.slice_z_levels = None
                 self.slice_layer_polygons = None
                 return

            # --- Generate G-code using the NEW function ---
            raw_gcode_string, generated_paths = slicing.generate_gcode_with_infill(
                z_levels, layer_polygons, # Pass polygons
                feedrate=feed,
                extrusion_per_mm=ext_mm,
                travel_feed=travel,
                infill_spacing=infill_space, # Pass infill spacing
                infill_angle=infill_angle    # Pass infill angle
            )
                # --- End Generate G-code ---
            self.generated_slice_paths = generated_paths
            with open(raw_gcode_path, "w") as f:
                f.write(raw_gcode_string)
            self.statusBar().showMessage(f"Wrote raw G-code to {os.path.basename(raw_gcode_path)}")
            QApplication.processEvents()

            if self.slice_optimize_cb.isChecked():
                self.statusBar().showMessage("Optimizing G-code...")
                QApplication.processEvents()
                raw_lines = raw_gcode_string.splitlines()
                opt_lines = gcode_optimizer.optimize_gcode(raw_lines) # Assuming optimizer still works
                with open(opt_gcode_path, "w") as f:
                    f.write("\n".join(opt_lines))
                self.statusBar().showMessage(f"Wrote optimized G-code to {os.path.basename(opt_gcode_path)}. Click 'Show Paths' to visualize.")
            else:
                self.statusBar().showMessage(f"Raw G-code saved. Optimization skipped. Click 'Show Paths' to visualize.")

        except Exception as e:
            self.statusBar().showMessage(f"Slicing Error: {e}")
            # Clear stored data on error
            self.slice_z_levels = None
            self.slice_layer_polygons = None # Clear polygons
            # Optionally print traceback for debugging
            import traceback
            traceback.print_exc()
        finally:
            if tmp_stl_file and os.path.exists(tmp_stl_file):
                try:
                    os.remove(tmp_stl_file)
                except OSError:
                    print(f"Warning: Could not delete temp file {tmp_stl_file}")

    def display_slice_paths(self):
        """Draws the stored slice paths in the 3D viewer, colored by height."""
        self.plotter.remove_actor(self.slice_paths_actor_name) # Clear previous paths

        # --- This function needs updating based on the new slicer output ---
        # Option 1: Visualize Polygon Outlines
        if self.slice_layer_polygons is None or self.slice_z_levels is None:
             self.statusBar().showMessage("No polygon slice data available.")
             return

        self.statusBar().showMessage("Visualizing polygon outlines (Infill not shown)...")
        QApplication.processEvents()

        all_points = []
        all_lines = []
        point_count = 0

        for z, polygons in zip(self.slice_z_levels, self.slice_layer_polygons):
             for poly in polygons:
                 # Exterior
                 exterior_coords = np.array(poly.exterior.coords)
                 if exterior_coords.shape[0] >= 2:
                     num_points = exterior_coords.shape[0]
                     z_coords = np.full((num_points, 1), z)
                     loop_xyz = np.hstack((exterior_coords, z_coords))
                     all_points.append(loop_xyz)
                     line_indices = np.arange(point_count, point_count + num_points)
                     lines_this_loop = np.vstack([np.full(num_points - 1, 2), line_indices[:-1], line_indices[1:]]).T.flatten()
                     all_lines.append(lines_this_loop)
                     point_count += num_points

                 # Interiors (Holes)
                 for interior in poly.interiors:
                     interior_coords = np.array(interior.coords)
                     if interior_coords.shape[0] >= 2:
                         num_points = interior_coords.shape[0]
                         z_coords = np.full((num_points, 1), z)
                         loop_xyz = np.hstack((interior_coords, z_coords))
                         all_points.append(loop_xyz)
                         line_indices = np.arange(point_count, point_count + num_points)
                         lines_this_loop = np.vstack([np.full(num_points - 1, 2), line_indices[:-1], line_indices[1:]]).T.flatten()
                         all_lines.append(lines_this_loop)
                         point_count += num_points

        if not all_points:
              self.statusBar().showMessage("No valid polygon outlines found.")
              return

        combined_points = np.vstack(all_points)
        combined_lines = np.concatenate(all_lines)
        z_scalars = combined_points[:, 2]

        line_polydata = pv.PolyData(combined_points)
        line_polydata.lines = combined_lines
        line_polydata['Z-Height'] = z_scalars

        scalar_bar_args = {
            'title': 'Z Height (mm)', 'vertical': True, 'position_x': 0.05,
            'position_y': 0.25, 'width': 0.1, 'height': 0.5,
            'title_font_size': 16, 'label_font_size': 12
        }
        self.plotter.add_mesh(
            line_polydata, scalars='Z-Height', cmap='viridis',
            line_width=2, # Make outlines visible
            scalar_bar_args=scalar_bar_args,
            name=self.slice_paths_actor_name
        )
        self.statusBar().showMessage("Polygon outlines visualized (colored by height).")



    def load_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl)")
        if not fn: return

        try:
            self.base_mesh = pv.read(fn)
            if hasattr(self.base_mesh, 'n_faces'):
                 self.base_mesh = self.base_mesh.triangulate()
            else:
                 raise TypeError(f"Loaded object is not a mesh: {type(self.base_mesh)}")
        except Exception as e:
            self.statusBar().showMessage(f"Error loading STL: {e}")
            self.base_mesh = None
            return

        self.original_mesh = self.base_mesh.copy()
        self.deformed_mesh  = None
        self.overhang_faces = None
        self.slice_z_levels = None
        self.slice_layers = None
        # No need to remove slice paths actor here, update_display clears actors

        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)

        self.print_bed = deformer.determine_print_bed(self.original_mesh)
        self.mesh_vis_cb.setChecked(True) # Ensure mesh is visible when loading new model
        self.update_display()
        self.status.showMessage(f"Loaded {os.path.basename(fn)}")


    def apply_manual_rotation(self):
        if not hasattr(self, "base_mesh") or self.base_mesh is None:
            return

        angle_x = self.rot_x_sb.value()
        angle_y = self.rot_y_sb.value()
        angle_z = self.rot_z_sb.value()

        m = self.base_mesh.copy()
        m.rotate_x(angle_x, inplace=True)
        m.rotate_y(angle_y, inplace=True)
        m.rotate_z(angle_z, inplace=True)

        self.original_mesh  = m
        self.deformed_mesh  = None
        self.overhang_faces = None
        self.print_bed     = deformer.determine_print_bed(m)
        self.slice_z_levels = None
        self.slice_layers = None
        # No need to remove slice paths actor here, update_display clears actors

        self.update_display()
        self.status.showMessage(
            f"Manual rot: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°"
        )


    def update_display(self):
        """Unified function to redraw the scene based on current state."""
        # --- Clear existing actors ---
        # Remove specific actors we manage here. Keep axes, scalar bars etc.
        actors_to_remove = ["main_mesh", "original_wireframe", "overhangs",
                            "print_bed_plane", "status_text", "overhang_text"]
        # Also remove slice paths if they exist
        if self.plotter.renderer.actors.get(self.slice_paths_actor_name):
             actors_to_remove.append(self.slice_paths_actor_name)

        for name in actors_to_remove:
            self.plotter.remove_actor(name, render=False) # Delay render

        # --- Determine which mesh to potentially show ---
        mesh_to_show = self.deformed_mesh if self.deformed_mesh is not None else self.original_mesh

        # --- Add Main Mesh (Original or Deformed) ---
        if mesh_to_show and self.mesh_vis_cb.isChecked(): # Check visibility checkbox
            style = 'wireframe' if self.wf_cb.isChecked() else 'surface'
            color = 'cyan' if self.deformed_mesh is not None else 'white'
            opacity = 1.0 # Default opacity

            # Add the main mesh
            self.plotter.add_mesh(mesh_to_show,
                                  style=style,
                                  show_edges=self.ed_cb.isChecked(),
                                  color=color,
                                  opacity=opacity,
                                  name="main_mesh") # Give it a name

            # If deformed, add original as wireframe for comparison
            if self.deformed_mesh is not None and self.original_mesh is not None:
                 self.plotter.add_mesh(self.original_mesh,
                                       style='wireframe',
                                       color='white',
                                       opacity=0.2,
                                       name="original_wireframe")
        elif not mesh_to_show:
             # No mesh loaded
             self.plotter.add_text("Load a model to begin", font_size=16,
                                   position="upper_left", name="status_text")


        # --- Add Overhangs ---
        if self.overhang_faces is not None and self.overhang_faces.size > 0:
             mesh_over = self.original_mesh.extract_cells(self.overhang_faces)
             self.plotter.add_mesh(mesh_over, color='red', opacity=0.8, name="overhangs")
             self.plotter.add_text(f"{self.overhang_faces.size} overhangs", position='upper_left', font_size=14, name="overhang_text")


        # --- Add Print Bed ---
        if self.bed_cb.isChecked():
            self._draw_bed() # This adds the 'print_bed_plane' actor


        # --- Add Slice Paths (if they exist) ---
        # Slice paths are only added/removed by display_slice_paths and slice_model
        # If you want them to persist across updates, you'd need to re-add them here
        # based on self.slice_layers, but the current logic is simpler.
        # We *do* need to re-add the actor if it exists but was cleared above.
        if self.slice_layers is not None:
             # Re-run the display logic to add them back if they should be visible
             # This is slightly inefficient but ensures they reappear after clear_actors
             self.display_slice_paths()


        # --- Final Render ---
        self.plotter.render() # Render once after all updates


    def _draw_bed(self):
        """Adds the print bed plane actor."""
        if self.print_bed is None:
            if self.original_mesh:
                self.print_bed = deformer.determine_print_bed(self.original_mesh)
            else:
                return

        plane_normal, plane_point = self.print_bed
        bed_z = plane_point[2]
        fixed_x_size = 200.0
        fixed_y_size = 200.0
        bed_center = (0.0, 0.0, bed_z)

        plane = pv.Plane(
            center=bed_center,
            direction=plane_normal,
            i_size=fixed_x_size,
            j_size=fixed_y_size
        )
        self.plotter.add_mesh(plane, name='print_bed_plane', color='lightblue', opacity=0.3)


    def set_orientation_preset(self):
        """Rotates the mesh based on the selected preset orientation."""
        if not hasattr(self, "base_mesh") or self.base_mesh is None:
            return

        preset = self.orient_combo.currentText()
        m = self.base_mesh.copy()

        if preset == "Bottom (XY)": pass
        elif preset == "Front (XZ)": m.rotate_x(90, inplace=True)
        elif preset == "Side (YZ)": m.rotate_y(-90, inplace=True)
        elif preset == "Top (-XY)": m.rotate_x(180, inplace=True)
        elif preset == "Back (-XZ)": m.rotate_x(-90, inplace=True)
        elif preset == "Other (-YZ)": m.rotate_y(90, inplace=True)

        self.original_mesh  = m
        self.deformed_mesh  = None
        self.overhang_faces = None
        self.print_bed     = deformer.determine_print_bed(m)
        self.slice_z_levels = None
        self.slice_layers = None

        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)

        self.update_display()
        self.status.showMessage(f"Orientation set to: {preset}")


    def detect_overhangs(self):
        if self.original_mesh is None:
            return

        mesh_to_check = self.original_mesh
        current_print_bed = self.print_bed

        angle = self.ang_sb.value()
        support_dist = self.sup_sb.value()
        use_ray      = self.ray_cb.isChecked()

        faces = deformer.detect_overhangs(
            mesh_to_check,
            angle_threshold=angle,
            print_bed=current_print_bed
        )

        if use_ray and faces.size > 0:
            centers = mesh_to_check.cell_centers().points
            filtered = []
            ray_direction = -np.array(current_print_bed[0])

            for f in faces:
                c = centers[f]
                start_point = c + ray_direction * 1e-4
                hits, _ = mesh_to_check.ray_trace(
                    start_point, ray_direction, first_point=True
                )
                if len(hits) == 0 or np.linalg.norm(hits[0] - c) > support_dist:
                    filtered.append(f)
            faces = np.array(filtered, dtype=int)

        self.overhang_faces = faces
        self.update_display()

        if faces.size > 0:
            self.status.showMessage(f"Detected {faces.size} overhangs")
        else:
            self.status.showMessage("No overhangs detected")


    def deform_model(self):
        if self.original_mesh is None:
            self.statusBar().showMessage("No model loaded")
            return

        mesh_to_deform = self.original_mesh
        current_print_bed = self.print_bed

        angle   = self.ang_sb.value()
        support = self.sup_sb.value()
        use_ray = self.ray_cb.isChecked()
        max_iter= 50

        self.statusBar().showMessage("Deforming mesh...")
        QApplication.processEvents()

        try:
            self.deformed_mesh, modified_vertices = deformer.deform_mesh(
                mesh_to_deform,
                angle_threshold   = angle,
                print_bed         = current_print_bed,
                support_distance  = support,
                use_ray           = use_ray,
                max_iterations    = max_iter
            )
            self.update_display()
            self.statusBar().showMessage(f"Deformed {len(modified_vertices)} vertices")

        except Exception as e:
             self.statusBar().showMessage(f"Deformation Error: {e}")
             self.deformed_mesh = None
             self.update_display()


    def reset_model(self):
        if not hasattr(self, "base_mesh") or self.base_mesh is None:
            self.statusBar().showMessage("No base model loaded to reset to.")
            return

        self.original_mesh = self.base_mesh.copy()
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = deformer.determine_print_bed(self.original_mesh)
        self.slice_z_levels = None
        self.slice_layers = None

        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)

        self.mesh_vis_cb.setChecked(True) # Ensure mesh is visible on reset
        self.update_display()
        self.status.showMessage("Model reset to original loaded state")


    def export_original(self):
        mesh_to_save = self.original_mesh
        if not mesh_to_save:
            self.statusBar().showMessage("No original mesh data to export.")
            return
        fn, _ = QFileDialog.getSaveFileName(self,
            "Save Original Mesh", "", "STL Files (*.stl)")
        if fn:
            try:
                mesh_to_save.save(fn)
                self.status.showMessage(f"Saved original mesh to {fn}")
            except Exception as e:
                self.status.showMessage(f"Error saving original mesh: {e}")

    def export_deformed(self):
        mesh_to_save = self.deformed_mesh
        if not mesh_to_save:
            self.statusBar().showMessage("No deformed mesh data to export.")
            return
        fn, _ = QFileDialog.getSaveFileName(self,
            "Save Deformed Mesh", "", "STL Files (*.stl)")
        if fn:
            try:
                mesh_to_save.save(fn)
                self.status.showMessage(f"Saved deformed mesh to {fn}")
            except Exception as e:
                self.status.showMessage(f"Error saving deformed mesh: {e}")


if __name__=="__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
