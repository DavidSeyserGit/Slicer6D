#!/usr/bin/env python3
import sys
import os
import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame, QGridLayout,
    QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QLabel, QSplitter, QStatusBar, QScrollArea, QSlider
)
from PyQt5.QtCore import Qt
from pyvistaqt import QtInteractor # Handles mouse/touchpad interaction
import tempfile
import traceback

# Import our modules
from deformer import deformer
from slicing import slicing
from slicing import gcode_optimizer


params = {
        "max_overhang_deg": 10.0,
        "neighbour_loss_weight": 30.0,
        "rotation_multiplier": 1.5,
        "initial_rotation_field_smoothing": 20,
        "set_initial_rotation_to_zero": False,
        "steep_overhang_compensation": True,
        "max_pos_rotation_deg": 45.0, # Limit max rotation more reasonably?
        "max_neg_rotation_deg": -45.0,
        "optimization_iterations": 10, # Fewer iterations for example
        "deformation_iterations": 10, # Fewer iterations for example
        "save_gifs": False, # Set to True to generate GIFs (can be slow)
        "model_name": "pi_3mm_deformed_example"
    }

# Constants for actor names
ACTOR_MAIN_MESH = "main_mesh"
ACTOR_ORIG_WIREFRAME = "original_wireframe" # Actor for the faint original mesh comparison
ACTOR_OVERHANGS = "overhangs"
ACTOR_PRINT_BED = "print_bed_plane"
ACTOR_STATUS_TEXT = "status_text"
ACTOR_OVERHANG_TEXT = "overhang_text"
ACTOR_FULL_PATHS = "slice_paths"
ACTOR_LAYER_VIEW_PATHS = "layer_view_paths"
ACTOR_LAYER_VIEW_MESH = "layer_view_mesh"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slicer6D")
        self.resize(1300, 900)

        # Model data
        self.original_mesh = None
        self.deformed_mesh = None
        self.overhang_faces = None
        self.print_bed = None
        self.base_mesh = None

        # Slice data
        self.slice_z_levels = None
        self.slice_layer_polygons = None
        self.generated_slice_paths = None
        self.max_layer_index = -1
        self.is_layer_view_active = False

        # --- Central Widget & Layout ---
        central = QWidget()
        self.setCentralWidget(central)
        hl = QHBoxLayout(central)

        # --- Splitter: Controls | Viewer ---
        splitter = QSplitter(Qt.Horizontal)
        hl.addWidget(splitter)

        # --- Control Panel Setup ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(350)
        scroll_area.setMaximumWidth(450)
        scroll_content = QWidget()
        vctrl = QVBoxLayout(scroll_content)

        # Build control groups
        self._build_file_group(vctrl)
        self._build_view_group(vctrl)
        self._build_orientation_group(vctrl)
        self._build_deform_group(vctrl)
        self._build_slice_group(vctrl)

        vctrl.addStretch()
        scroll_content.setLayout(vctrl)
        scroll_area.setWidget(scroll_content)
        splitter.addWidget(scroll_area)

        # --- Viewer Frame ---
        viewer_frame = QFrame()
        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        # --- QtInteractor Setup ---
        # This widget provides the 3D view and handles mouse/touchpad input
        # for rotation (LMB drag), zoom (Scroll/RMB drag/Pinch), and pan (MMB drag/Shift+LMB drag)
        self.plotter = QtInteractor(viewer_frame)
        # --- End QtInteractor Setup ---
        viewer_layout.addWidget(self.plotter)
        viewer_frame.setLayout(viewer_layout)
        splitter.addWidget(viewer_frame)

        splitter.setSizes([400, 900])

        # --- Initial Plotter Setup ---
        self.plotter.set_background([0.2, 0.2, 0.2])
        self.plotter.add_axes(interactive=True, line_width=3)
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.plotter.add_text("Load a model to begin", font_size=16,
                              position="upper_left", name=ACTOR_STATUS_TEXT, color='white')
        self.status.showMessage("Ready. Load STL model to start.") # Initial status message

    def _build_file_group(self, parent_layout):
        gb = QGroupBox("File")
        l = QVBoxLayout(gb)
        btn_load = QPushButton("Load STL")
        btn_load.clicked.connect(self.load_model)
        l.addWidget(btn_load)
        hb_export = QHBoxLayout()
        btn_export_orig = QPushButton("Export Original")
        btn_export_orig.clicked.connect(self.export_original)
        btn_export_def = QPushButton("Export Deformed")
        btn_export_def.clicked.connect(self.export_deformed)
        hb_export.addWidget(btn_export_orig)
        hb_export.addWidget(btn_export_def)
        l.addLayout(hb_export)
        parent_layout.addWidget(gb)

    def _build_view_group(self, parent_layout):
        gb = QGroupBox("View")
        g = QGridLayout(gb)
        bxy = QPushButton("Top (XY)"); bxy.clicked.connect(lambda: self.plotter.view_xy())
        bxz = QPushButton("Front (XZ)"); bxz.clicked.connect(lambda: self.plotter.view_xz())
        byz = QPushButton("Side (YZ)"); byz.clicked.connect(lambda: self.plotter.view_yz())
        bis = QPushButton("Isometric"); bis.clicked.connect(lambda: self.plotter.view_isometric())
        g.addWidget(bxy, 0, 0); g.addWidget(bxz, 0, 1)
        g.addWidget(byz, 1, 0); g.addWidget(bis, 1, 1)
        self.mesh_vis_cb = QCheckBox("Show Main Mesh")
        self.mesh_vis_cb.setChecked(True)
        self.mesh_vis_cb.stateChanged.connect(self.update_display)
        g.addWidget(self.mesh_vis_cb, 2, 0, 1, 2)
        self.wf_cb = QCheckBox("Wireframe")
        self.wf_cb.stateChanged.connect(self.update_display)
        self.ed_cb = QCheckBox("Show Edges")
        self.ed_cb.setChecked(True)
        self.ed_cb.stateChanged.connect(self.update_display)
        g.addWidget(self.wf_cb, 3, 0); g.addWidget(self.ed_cb, 3, 1)
        parent_layout.addWidget(gb)

    def _build_orientation_group(self, parent_layout):
        gb = QGroupBox("Print Orientation")
        layout = QVBoxLayout(gb)
        h_preset = QHBoxLayout()
        h_preset.addWidget(QLabel("Base:"))
        self.orient_combo = QComboBox()
        self.orient_combo.addItems(["Bottom (XY)", "Front (XZ)", "Side (YZ)", "Top (-XY)", "Back (-XZ)", "Other (-YZ)"])
        h_preset.addWidget(self.orient_combo)
        btn_set_preset = QPushButton("Set Preset")
        btn_set_preset.clicked.connect(self.set_orientation_preset)
        h_preset.addWidget(btn_set_preset)
        layout.addLayout(h_preset)
        rot_grid = QGridLayout()
        rot_grid.addWidget(QLabel("Rotate X:"), 0, 0)
        self.rot_x_sb = QDoubleSpinBox()
        self.rot_x_sb.setRange(-180, 180); self.rot_x_sb.setSingleStep(5); self.rot_x_sb.setSuffix("°")
        self.rot_x_sb.valueChanged.connect(self.apply_manual_rotation)
        rot_grid.addWidget(self.rot_x_sb, 0, 1)
        rot_grid.addWidget(QLabel("Rotate Y:"), 1, 0)
        self.rot_y_sb = QDoubleSpinBox()
        self.rot_y_sb.setRange(-180, 180); self.rot_y_sb.setSingleStep(5); self.rot_y_sb.setSuffix("°")
        self.rot_y_sb.valueChanged.connect(self.apply_manual_rotation)
        rot_grid.addWidget(self.rot_y_sb, 1, 1)
        rot_grid.addWidget(QLabel("Rotate Z:"), 2, 0)
        self.rot_z_sb = QDoubleSpinBox()
        self.rot_z_sb.setRange(-180, 180); self.rot_z_sb.setSingleStep(5); self.rot_z_sb.setSuffix("°")
        self.rot_z_sb.valueChanged.connect(self.apply_manual_rotation)
        rot_grid.addWidget(self.rot_z_sb, 2, 1)
        layout.addLayout(rot_grid)
        self.bed_cb = QCheckBox("Show Print Bed")
        self.bed_cb.setChecked(True)
        self.bed_cb.stateChanged.connect(self.update_display)
        layout.addWidget(self.bed_cb)
        parent_layout.addWidget(gb)

    def _build_deform_group(self, parent_layout):
        gb = QGroupBox("Deformation")
        l = QVBoxLayout(gb)
        hb = QHBoxLayout()
        b1 = QPushButton("Deform Overhangs")
        b1.clicked.connect(self.deform_model)
        b2 = QPushButton("Reset Model")
        b2.clicked.connect(self.reset_model)
        hb.addWidget(b1); hb.addWidget(b2)
        l.addLayout(hb)
        parent_layout.addWidget(gb)

    def _build_slice_group(self, parent_layout):
        gb = QGroupBox("Slicing & Layer View")
        l = QVBoxLayout(gb)
        grid = QGridLayout()
        grid.addWidget(QLabel("Layer Height:"), 0, 0)
        self.slice_layer_sb = QDoubleSpinBox()
        self.slice_layer_sb.setRange(0.05, 5.0); self.slice_layer_sb.setSingleStep(0.05)
        self.slice_layer_sb.setValue(0.2); self.slice_layer_sb.setSuffix(" mm")
        grid.addWidget(self.slice_layer_sb, 0, 1)
        grid.addWidget(QLabel("Extrusion (E/mm):"), 1, 0)
        self.slice_extrude_sb = QDoubleSpinBox()
        self.slice_extrude_sb.setRange(0.01, 1.0); self.slice_extrude_sb.setSingleStep(0.01)
        self.slice_extrude_sb.setValue(0.05)
        grid.addWidget(self.slice_extrude_sb, 1, 1)
        grid.addWidget(QLabel("Print Feed (mm/min):"), 2, 0)
        self.slice_feed_sb = QSpinBox()
        self.slice_feed_sb.setRange(100, 10000); self.slice_feed_sb.setValue(1500)
        self.slice_feed_sb.setSingleStep(100)
        grid.addWidget(self.slice_feed_sb, 2, 1)
        grid.addWidget(QLabel("Travel Feed (mm/min):"), 3, 0)
        self.slice_travel_sb = QSpinBox()
        self.slice_travel_sb.setRange(100, 15000); self.slice_travel_sb.setValue(3000)
        self.slice_travel_sb.setSingleStep(100)
        grid.addWidget(self.slice_travel_sb, 3, 1)
        grid.addWidget(QLabel("Infill Spacing:"), 4, 0)
        self.slice_infill_spacing_sb = QDoubleSpinBox()
        self.slice_infill_spacing_sb.setRange(0.1, 20.0)
        self.slice_infill_spacing_sb.setSingleStep(0.5)
        self.slice_infill_spacing_sb.setValue(2.0)
        self.slice_infill_spacing_sb.setSuffix(" mm")
        grid.addWidget(self.slice_infill_spacing_sb, 4, 1)
        grid.addWidget(QLabel("Infill Angle:"), 5, 0)
        self.slice_infill_angle_sb = QSpinBox()
        self.slice_infill_angle_sb.setRange(0, 179)
        self.slice_infill_angle_sb.setValue(45)
        self.slice_infill_angle_sb.setSuffix(" °")
        grid.addWidget(self.slice_infill_angle_sb, 5, 1)
        l.addLayout(grid)
        self.slice_deformed_cb = QCheckBox("Slice Deformed Mesh (if available)")
        l.addWidget(self.slice_deformed_cb)
        self.slice_optimize_cb = QCheckBox("Optimize G‑code (basic)")
        self.slice_optimize_cb.setChecked(True)
        l.addWidget(self.slice_optimize_cb)
        btn_layout = QHBoxLayout()
        slice_btn = QPushButton("Slice to G‑code")
        slice_btn.setMinimumHeight(36)
        slice_btn.clicked.connect(self.slice_model)
        btn_layout.addWidget(slice_btn)
        vis_btn = QPushButton("Show Full Toolpaths")
        vis_btn.setMinimumHeight(36)
        vis_btn.clicked.connect(self.display_full_slice_paths)
        btn_layout.addWidget(vis_btn)
        l.addLayout(btn_layout)
        l.addWidget(QLabel("Layer View:"))
        self.layer_slider_label = QLabel("Layer: N/A")
        l.addWidget(self.layer_slider_label)
        self.layer_slider = QSlider(Qt.Horizontal)
        self.layer_slider.setMinimum(0)
        self.layer_slider.setMaximum(0)
        self.layer_slider.setEnabled(False)
        self.layer_slider.valueChanged.connect(self._update_layer_view)
        l.addWidget(self.layer_slider)
        parent_layout.addWidget(gb)

    # --- Core Logic Methods ---

    def _clear_layer_view(self):
        """Removes layer view specific actors and resets state."""
        self.plotter.remove_actor(ACTOR_LAYER_VIEW_PATHS, render=False)
        self.plotter.remove_actor(ACTOR_LAYER_VIEW_MESH, render=False)
        # Also ensure the original wireframe comparison isn't stuck on
        self.plotter.remove_actor(ACTOR_ORIG_WIREFRAME, render=False)
        self.is_layer_view_active = False

    def _reset_slice_data(self):
        """Clears all slicing-related data and UI elements."""
        self.generated_slice_paths = None
        self.slice_z_levels = None
        self.slice_layer_polygons = None
        self.max_layer_index = -1
        self.layer_slider.setEnabled(False)
        self.layer_slider.setMaximum(0)
        # Block signals temporarily while resetting slider value to avoid triggering _update_layer_view
        self.layer_slider.blockSignals(True)
        self.layer_slider.setValue(0)
        self.layer_slider.blockSignals(False)
        self.layer_slider_label.setText("Layer: N/A")
        self._clear_layer_view()
        self.plotter.remove_actor(ACTOR_FULL_PATHS, render=False)

    # --- load_model, apply_manual_rotation, set_orientation_preset, ---
    # --- detect_overhangs, deform_model, reset_model ---
    def load_model(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Open STL", "", "STL Files (*.stl *.STL)")
        if not fn: return
        self.status.showMessage(f"Loading {os.path.basename(fn)}...")
        QApplication.processEvents()
        try:
            mesh = pv.read(fn)
            if isinstance(mesh, pv.PolyData):
                 if mesh.n_cells == 0: raise ValueError("Mesh has no faces.")
                 faces_conn = mesh.faces
                 if faces_conn.size > 0:
                     expected_size = faces_conn[0] + 1
                     if faces_conn.size % expected_size != 0 or not np.all(faces_conn[::expected_size] == 3):
                          print("Non-triangular faces detected, attempting triangulation.")
                          self.base_mesh = mesh.triangulate()
                     else: self.base_mesh = mesh
                 else: self.base_mesh = mesh.triangulate()
            elif isinstance(mesh, (pv.UnstructuredGrid, pv.MultiBlock)):
                 print("Loaded complex type, extracting/merging surface geometry.")
                 self.base_mesh = mesh.extract_geometry().triangulate()
            else: raise TypeError(f"Unsupported mesh type: {type(mesh)}")
            if not isinstance(self.base_mesh, pv.PolyData) or self.base_mesh.n_cells == 0:
                 raise ValueError("Failed to obtain a valid surface mesh.")
        except Exception as e:
            self.status.showMessage(f"Error loading/processing STL: {e}")
            traceback.print_exc()
            self.base_mesh = None; self.original_mesh = None; self.deformed_mesh = None
            self.overhang_faces = None; self.print_bed = None
            self._reset_slice_data()
            self.update_display(); return
        
        self.original_mesh = self.base_mesh.copy()
        self.deformed_mesh  = None; self.overhang_faces = None
        self._reset_slice_data()
        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)
        try: self.print_bed = deformer.determine_print_bed(self.original_mesh)
        except Exception as e:
            self.status.showMessage(f"Warning: Could not determine print bed: {e}")
            center = self.original_mesh.center; z_min = self.original_mesh.bounds[4]
            self.print_bed = ([0, 0, 1], [center[0], center[1], z_min])
        self.mesh_vis_cb.setChecked(True)
        self.update_display()
        self.plotter.view_isometric(); self.plotter.reset_camera(render=True)
        self.status.showMessage(f"Loaded: {os.path.basename(fn)} ({self.original_mesh.n_cells} faces)")

    def apply_manual_rotation(self):
        if not hasattr(self, "base_mesh") or self.base_mesh is None: return
        angle_x = self.rot_x_sb.value(); angle_y = self.rot_y_sb.value(); angle_z = self.rot_z_sb.value()
        m = self.base_mesh.copy(); center = m.center
        m.rotate_x(angle_x, point=center, inplace=True); m.rotate_y(angle_y, point=center, inplace=True); m.rotate_z(angle_z, point=center, inplace=True)
        self.original_mesh  = m; self.deformed_mesh  = None; self.overhang_faces = None
        self._reset_slice_data()
        try: self.print_bed = deformer.determine_print_bed(m)
        except Exception as e:
             self.status.showMessage(f"Warning: Could not determine print bed after rotation: {e}")
             center = m.center; z_min = m.bounds[4]; self.print_bed = ([0, 0, 1], [center[0], center[1], z_min])
        self.update_display()
        self.status.showMessage(f"Manual Rotation: X={angle_x:.1f}°, Y={angle_y:.1f}°, Z={angle_z:.1f}°")

    def set_orientation_preset(self):
        if not hasattr(self, "base_mesh") or self.base_mesh is None: return
        preset = self.orient_combo.currentText(); m = self.base_mesh.copy(); center = m.center
        if preset == "Bottom (XY)": pass
        elif preset == "Front (XZ)": m.rotate_x(90, point=center, inplace=True)
        elif preset == "Side (YZ)": m.rotate_y(-90, point=center, inplace=True)
        elif preset == "Top (-XY)": m.rotate_x(180, point=center, inplace=True)
        elif preset == "Back (-XZ)": m.rotate_x(-90, point=center, inplace=True)
        elif preset == "Other (-YZ)": m.rotate_y(90, point=center, inplace=True)
        self.original_mesh  = m; self.deformed_mesh  = None; self.overhang_faces = None
        self._reset_slice_data()
        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)
        try: self.print_bed = deformer.determine_print_bed(m)
        except Exception as e:
             self.status.showMessage(f"Warning: Could not determine print bed after preset: {e}")
             center = m.center; z_min = m.bounds[4]; self.print_bed = ([0, 0, 1], [center[0], center[1], z_min])
        self.update_display()
        self.status.showMessage(f"Orientation set to: {preset}")

    def deform_model(self):
        if self.original_mesh is None: self.status.showMessage("Load a model first."); return
        if self.print_bed is None: self.status.showMessage("Print bed not determined."); return

        #loaded from the load_model
        mesh_to_deform = self.original_mesh; current_print_bed = self.print_bed
        self.status.showMessage("Deforming mesh..."); QApplication.processEvents()
        try:
            deformed_result, modified_vertices = deformer.deform_mesh(
                mesh_to_deform, **params)
            self.deformed_mesh = deformed_result; self.overhang_faces = None
            self._reset_slice_data()
            self.update_display()
            self.status.showMessage(f"Deformation complete. Modified {len(modified_vertices)} vertices.")
        except Exception as e:
             self.status.showMessage(f"Deformation Error: {e}"); traceback.print_exc()
             self.deformed_mesh = None; self._reset_slice_data(); self.update_display()

    def reset_model(self):
        if not hasattr(self, "base_mesh") or self.base_mesh is None: self.status.showMessage("No base model loaded."); return
        self.original_mesh = self.base_mesh.copy(); self.deformed_mesh = None; self.overhang_faces = None
        self._reset_slice_data()
        self.rot_x_sb.blockSignals(True); self.rot_x_sb.setValue(0); self.rot_x_sb.blockSignals(False)
        self.rot_y_sb.blockSignals(True); self.rot_y_sb.setValue(0); self.rot_y_sb.blockSignals(False)
        self.rot_z_sb.blockSignals(True); self.rot_z_sb.setValue(0); self.rot_z_sb.blockSignals(False)
        try: self.print_bed = deformer.determine_print_bed(self.original_mesh)
        except Exception as e:
            self.status.showMessage(f"Warning: Could not determine print bed on reset: {e}")
            center = self.original_mesh.center; z_min = self.original_mesh.bounds[4]
            self.print_bed = ([0, 0, 1], [center[0], center[1], z_min])
        self.mesh_vis_cb.setChecked(True); self.update_display()
        self.status.showMessage("Model reset to original loaded state.")

    def slice_model(self):
        """Slices the model, generates G-code, enables layer slider."""
        self._reset_slice_data() # Clear previous results first

        mesh_to_slice = None; source_name = ""
        if self.slice_deformed_cb.isChecked() and self.deformed_mesh is not None:
            mesh_to_slice = self.deformed_mesh; source_name = "deformed"
        elif self.original_mesh is not None:
            mesh_to_slice = self.original_mesh; source_name = "original"
        else: self.status.showMessage("No mesh loaded or selected to slice."); return

        fn, _ = QFileDialog.getSaveFileName(self, "Save G‑code Base Name", "", "G‑code Files (*.gcode)")
        if not fn: return

        if not fn.lower().endswith('.gcode'): fn += '.gcode'
        base_path = os.path.splitext(fn)[0]
        raw_gcode_path = base_path + "_raw.gcode"
        opt_gcode_path = base_path + "_opt.gcode"

        tmp_stl_file = None
        try:
            self.status.showMessage(f"Preparing {source_name} mesh for slicing..."); QApplication.processEvents()
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_stl_file = tmp.name; mesh_to_slice.save(tmp_stl_file)

            self.status.showMessage(f"Slicing {source_name} mesh..."); QApplication.processEvents()
            layer_h = self.slice_layer_sb.value(); ext_mm = self.slice_extrude_sb.value()
            feed = self.slice_feed_sb.value(); travel = self.slice_travel_sb.value()
            infill_space = self.slice_infill_spacing_sb.value(); infill_angle = self.slice_infill_angle_sb.value()

            z_levels, layer_polygons = slicing.slice_mesh_to_polygons(tmp_stl_file, layer_h)
            if not z_levels or not layer_polygons or all(not lp for lp in layer_polygons):
                 self.status.showMessage("Slicing completed, but no valid layers found.")
                 self._reset_slice_data(); return

            self.slice_z_levels = z_levels; self.slice_layer_polygons = layer_polygons
            self.max_layer_index = len(z_levels) - 1

            self.status.showMessage("Generating G-code with infill..."); QApplication.processEvents()
            raw_gcode_string, generated_paths = slicing.generate_gcode_with_infill(
                z_levels, layer_polygons, feedrate=feed, extrusion_per_mm=ext_mm,
                travel_feed=travel, infill_spacing=infill_space, infill_angle=infill_angle)
            self.generated_slice_paths = generated_paths

            with open(raw_gcode_path, "w") as f: f.write(raw_gcode_string)
            self.status.showMessage(f"Wrote raw G-code to {os.path.basename(raw_gcode_path)}")
            QApplication.processEvents()

            if self.slice_optimize_cb.isChecked():
                self.status.showMessage("Optimizing G-code..."); QApplication.processEvents()
                try:
                    raw_lines = raw_gcode_string.splitlines()
                    if hasattr(gcode_optimizer, 'optimize_gcode'):
                        opt_lines = gcode_optimizer.optimize_gcode(raw_lines)
                        with open(opt_gcode_path, "w") as f: f.write("\n".join(opt_lines))
                        self.status.showMessage(f"Wrote optimized G-code to {os.path.basename(opt_gcode_path)}.")
                    else: self.status.showMessage("G-code optimizer function not found. Skipping.")
                except Exception as opt_e:
                    self.status.showMessage(f"G-code optimization failed: {opt_e}"); traceback.print_exc()
            else: self.status.showMessage(f"Raw G-code saved. Optimization skipped.")

            if self.max_layer_index >= 0:
                self.layer_slider.setEnabled(True)
                self.layer_slider.setMaximum(self.max_layer_index)
                # Set slider value without triggering update yet
                self.layer_slider.blockSignals(True)
                self.layer_slider.setValue(self.max_layer_index)
                self.layer_slider.blockSignals(False)
                # Now trigger the update to show the top layer
                self._update_layer_view()
                self.status.showMessage(f"Slice complete. Use slider for layer view or click 'Show Full Toolpaths'.")
            else: self.status.showMessage("Slicing done, but no layers to view.")

        except Exception as e:
            self.status.showMessage(f"Slicing/G-code Error: {e}"); traceback.print_exc()
            self._reset_slice_data()
        finally:
            if tmp_stl_file and os.path.exists(tmp_stl_file):
                try: os.remove(tmp_stl_file)
                except OSError as ose: print(f"Warning: Could not delete temp file {tmp_stl_file}: {ose}")

    def display_full_slice_paths(self):
        """Draws ALL generated slice paths, hiding other mesh actors."""
        self._clear_layer_view() # Ensure layer view is off
        self.plotter.remove_actor(ACTOR_FULL_PATHS, render=False)

        if self.generated_slice_paths is None or not self.generated_slice_paths:
             self.status.showMessage("No generated toolpath data available."); return

        self.status.showMessage("Visualizing FULL toolpaths..."); QApplication.processEvents()

        all_points_list = []; all_lines_list = []; point_count = 0
        for path_segment in self.generated_slice_paths:
            if isinstance(path_segment, np.ndarray) and path_segment.ndim == 2 and path_segment.shape[1] == 3 and path_segment.shape[0] >= 2:
                num_points = path_segment.shape[0]; all_points_list.append(path_segment)
                line_indices = np.arange(point_count, point_count + num_points)
                lines_this_segment = np.vstack([np.full(num_points - 1, 2), line_indices[:-1], line_indices[1:]]).T.flatten()
                all_lines_list.append(lines_this_segment); point_count += num_points
            else: print(f"Warning: Skipping invalid path segment data in full visualization.")

        if not all_points_list: self.status.showMessage("No valid toolpath segments found."); return

        try:
            combined_points = np.vstack(all_points_list); combined_lines = np.concatenate(all_lines_list)
            if combined_points.size == 0 or combined_lines.size == 0: self.status.showMessage("Combined toolpath data is empty."); return

            line_polydata = pv.PolyData(combined_points); line_polydata.lines = combined_lines
            z_scalars = combined_points[:, 2]; line_polydata['Z-Height'] = z_scalars
            scalar_bar_args = { 'title': 'Z Height (mm)', 'vertical': True, 'position_x': 0.05, 'position_y': 0.25,
                                'width': 0.1, 'height': 0.5, 'title_font_size': 14, 'label_font_size': 10, 'color': 'white' }

            # --- Add FULL paths actor ---
            self.plotter.add_mesh( line_polydata, scalars='Z-Height', cmap='viridis', line_width=2,
                                   scalar_bar_args=scalar_bar_args, name=ACTOR_FULL_PATHS )

            # --- Hide main mesh actors when showing full paths ---
            self.plotter.remove_actor(ACTOR_MAIN_MESH, render=False)
            self.plotter.remove_actor(ACTOR_ORIG_WIREFRAME, render=False) # Ensure wireframe is also hidden
            self.plotter.remove_actor(ACTOR_LAYER_VIEW_MESH, render=False) # Ensure clipped mesh is hidden
            self.plotter.render()
            self.status.showMessage(f"Full toolpaths visualized ({len(self.generated_slice_paths)} segments).")

        except Exception as e:
             self.status.showMessage(f"Error creating full toolpath visualization: {e}"); traceback.print_exc()

    def _update_layer_view(self):
        """Updates paths and mesh clipping based on the layer slider."""
        if not self.layer_slider.isEnabled() or self.generated_slice_paths is None or self.slice_z_levels is None:
            # If called when not ready (e.g., during reset), just ensure view is clear
            self._clear_layer_view()
            self.plotter.render()
            return

        current_layer_index = self.layer_slider.value()
        if not (0 <= current_layer_index <= self.max_layer_index): return

        target_z = self.slice_z_levels[current_layer_index]
        self.is_layer_view_active = True
        self.layer_slider_label.setText(f"Layer: {current_layer_index + 1} / {self.max_layer_index + 1} (Z={target_z:.3f})")

        # --- Filter Paths ---
        filtered_paths_segments = []
        epsilon = 1e-5
        for path_segment in self.generated_slice_paths:
             if isinstance(path_segment, np.ndarray) and path_segment.ndim == 2 and path_segment.shape[1] == 3:
                 if np.max(path_segment[:, 2]) <= target_z + epsilon:
                     filtered_paths_segments.append(path_segment)

        # --- Clip Mesh ---
        mesh_to_clip = self.deformed_mesh if self.deformed_mesh is not None else self.original_mesh
        clipped_mesh = None
        if mesh_to_clip:
            try:
                clipped_mesh = mesh_to_clip.clip(normal=[0, 0, 1], origin=[0, 0, target_z], invert=False, inplace=False)
            except Exception as clip_e: print(f"Warning: Mesh clipping failed: {clip_e}")

        # --- Update Plotter ---
        # Remove actors specific to other views FIRST
        self.plotter.remove_actor(ACTOR_MAIN_MESH, render=False)
        self.plotter.remove_actor(ACTOR_ORIG_WIREFRAME, render=False) # Explicitly remove wireframe
        self.plotter.remove_actor(ACTOR_FULL_PATHS, render=False)
        self.plotter.remove_actor(ACTOR_LAYER_VIEW_PATHS, render=False) # Remove old layer paths
        self.plotter.remove_actor(ACTOR_LAYER_VIEW_MESH, render=False)  # Remove old layer mesh

        # Add Clipped Mesh Actor
        if clipped_mesh and clipped_mesh.n_points > 0:
             style = 'wireframe' if self.wf_cb.isChecked() else 'surface'
             # Use the color appropriate for the clipped mesh (original or deformed)
             color = 'cyan' if self.deformed_mesh is not None else 'lightgrey'
             self.plotter.add_mesh(clipped_mesh, style=style, color=color,
                                   show_edges=self.ed_cb.isChecked(), edge_color='gray',
                                   name=ACTOR_LAYER_VIEW_MESH)

        # Add Filtered Paths Actor
        if filtered_paths_segments:
            all_points_list = []; all_lines_list = []; point_count = 0
            for path_segment in filtered_paths_segments:
                num_points = path_segment.shape[0]; all_points_list.append(path_segment)
                line_indices = np.arange(point_count, point_count + num_points)
                lines_this_segment = np.vstack([np.full(num_points - 1, 2), line_indices[:-1], line_indices[1:]]).T.flatten()
                all_lines_list.append(lines_this_segment); point_count += num_points
            if all_points_list:
                try:
                    combined_points = np.vstack(all_points_list); combined_lines = np.concatenate(all_lines_list)
                    if combined_points.size > 0 and combined_lines.size > 0:
                        line_polydata = pv.PolyData(combined_points); line_polydata.lines = combined_lines
                        z_scalars = combined_points[:, 2]; line_polydata['Z-Height'] = z_scalars
                        self.plotter.add_mesh(line_polydata, scalars='Z-Height', cmap='viridis',
                                              line_width=2, name=ACTOR_LAYER_VIEW_PATHS)
                except Exception as path_build_e: print(f"Error building layer view paths: {path_build_e}")

        self.plotter.render()

    def update_display(self):
        """Unified display update, respects layer view state."""
        actors_to_remove = [
            ACTOR_MAIN_MESH, ACTOR_ORIG_WIREFRAME, ACTOR_OVERHANGS, ACTOR_PRINT_BED,
            ACTOR_STATUS_TEXT, ACTOR_OVERHANG_TEXT, ACTOR_FULL_PATHS,
            ACTOR_LAYER_VIEW_PATHS, ACTOR_LAYER_VIEW_MESH ]
        for name in actors_to_remove:
            actor = self.plotter.renderer.actors.get(name)
            if actor: self.plotter.remove_actor(actor, render=False)

        # --- Add elements based on state ---
        if not self.is_layer_view_active:
            # --- Normal View (Not Layer View) ---
            mesh_to_show = self.deformed_mesh if self.deformed_mesh is not None else self.original_mesh
            if mesh_to_show and self.mesh_vis_cb.isChecked():
                style = 'wireframe' if self.wf_cb.isChecked() else 'surface'
                color = 'cyan' if self.deformed_mesh is not None else 'lightgrey'
                self.plotter.add_mesh(mesh_to_show, style=style, show_edges=self.ed_cb.isChecked(),
                                      edge_color='gray', color=color, name=ACTOR_MAIN_MESH)
                # Show wireframe comparison ONLY if deformed mesh exists AND layer view is OFF
                if self.deformed_mesh is not None and self.original_mesh is not None:
                     self.plotter.add_mesh(self.original_mesh, style='wireframe', color='white',
                                           opacity=0.15, line_width=1, name=ACTOR_ORIG_WIREFRAME)
            elif not mesh_to_show:
                 self.plotter.add_text("Load a model to begin", font_size=16,
                                       position="upper_left", name=ACTOR_STATUS_TEXT, color='white')

            # Add Overhangs (if not in layer view)
            if self.overhang_faces is not None and self.overhang_faces.size > 0 and self.original_mesh:
                 try:
                     valid_indices = self.overhang_faces[self.overhang_faces < self.original_mesh.n_cells]
                     if valid_indices.size > 0:
                         mesh_over = self.original_mesh.extract_cells(valid_indices)
                         if mesh_over and mesh_over.n_cells > 0:
                             self.plotter.add_mesh(mesh_over, color='red', opacity=0.7, name=ACTOR_OVERHANGS)
                             self.plotter.add_text(f"{valid_indices.size} overhangs", position='upper_right',
                                                   font_size=12, name=ACTOR_OVERHANG_TEXT, color='red')
                 except Exception as e: print(f"Error displaying overhangs: {e}")

            # Re-add full paths ONLY if they exist AND layer view is NOT active
            if self.generated_slice_paths is not None:
                 self.display_full_slice_paths()

        elif self.is_layer_view_active:
            # --- Layer View Active ---
            # _update_layer_view handles adding ACTOR_LAYER_VIEW_MESH and ACTOR_LAYER_VIEW_PATHS
            # We might need to re-call it if view options (like wireframe) changed
            self._update_layer_view()

        # Add Print Bed (always show if checked)
        if self.bed_cb.isChecked():
            self._draw_bed()

        self.plotter.render() # Final render

    def _draw_bed(self):
        # (Identical to previous version)
        if self.print_bed is None:
            if self.original_mesh:
                try: self.print_bed = deformer.determine_print_bed(self.original_mesh)
                except Exception: return
            else: return
        try:
            plane_normal, plane_point = self.print_bed
            plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
            plane_point = np.array(plane_point)
            bed_x_size = 300.0; bed_y_size = 300.0
            bed_center = (self.original_mesh.center[0], self.original_mesh.center[1], plane_point[2]) if self.original_mesh else plane_point
            plane = pv.Plane(center=bed_center, direction=plane_normal, i_size=bed_x_size, j_size=bed_y_size, i_resolution=1, j_resolution=1)
            self.plotter.add_mesh(plane, name=ACTOR_PRINT_BED, color='lightblue', opacity=0.2)
        except Exception as e: print(f"Error drawing print bed: {e}")

    def export_original(self):
        # (Identical to previous version)
        mesh_to_save = self.original_mesh
        if not mesh_to_save: self.status.showMessage("No original mesh data."); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Original Mesh", "", "STL Files (*.stl)")
        if fn:
            try: mesh_to_save.save(fn); self.status.showMessage(f"Saved original mesh to {os.path.basename(fn)}")
            except Exception as e: self.status.showMessage(f"Error saving original: {e}"); traceback.print_exc()

    def export_deformed(self):
        # (Identical to previous version)
        mesh_to_save = self.deformed_mesh
        if not mesh_to_save: self.status.showMessage("No deformed mesh data."); return
        fn, _ = QFileDialog.getSaveFileName(self, "Save Deformed Mesh", "", "STL Files (*.stl)")
        if fn:
            try: mesh_to_save.save(fn); self.status.showMessage(f"Saved deformed mesh to {os.path.basename(fn)}")
            except Exception as e: self.status.showMessage(f"Error saving deformed: {e}"); traceback.print_exc()

# --- Main Application Execution ---
if __name__=="__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
