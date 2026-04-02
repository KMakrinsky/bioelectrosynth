import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLViewWidget import GLViewWidget
from .mouse_phantom import MousePhantom

class BioPhantomViewer(QtWidgets.QWidget):
    geometry_changed = QtCore.pyqtSignal(float, list, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('k')
        self.view.setCameraPosition(distance=18, elevation=25, azimuth=-90)
        self.view.installEventFilter(self)
        self.layout.addWidget(self.view, stretch=1)
        
        info_widget = QtWidgets.QWidget()
        info_widget.setStyleSheet("background-color: #2b2b2b; color: #eeeeee;")
        info_layout = QtWidgets.QHBoxLayout(info_widget)
        
        self.lbl_info = QtWidgets.QLabel("Drag green/yellow spheres to move electrodes.")
        self.lbl_info.setStyleSheet("font-weight: bold; font-size: 14px; color: #00ff00;")
        info_layout.addWidget(self.lbl_info)
        
        self.layout.addWidget(info_widget)

        self.phantom = MousePhantom()
        self.dragging_electrode = None 
        self._last_pos = None
        
        self.pos_w_local = np.array([2.0, 0.8, 0]) 
        self.pos_w_local = self.phantom.snap_to_surface(self.pos_w_local[0], self.pos_w_local[1])
        
        self.pos_c_local = np.array([-1.0, -0.5, 0])
        self.pos_c_local = self.phantom.snap_to_surface(self.pos_c_local[0], self.pos_c_local[1])
        
        self._init_scene()
        self._update_visuals()
        
    def _init_scene(self):
        g = gl.GLGridItem()
        g.scale(1, 1, 1)
        g.setColor((50, 50, 50, 255))
        self.view.addItem(g)
        
        meshes = self.phantom.generate_mesh_data(hulling=False, u_res=30, v_res=20) # Use lower res for realtime performance
        for m in meshes:
            x, y, z = m['x'], m['y'], m['z']
            nx, ny, nz = m['nx'], m['ny'], m['nz'] # Extract pre-calculated normals
            
            rows, cols = x.shape
            verts = []
            norms = []
            faces = []
            colors = []
            
            for r in range(rows):
                for c in range(cols):
                    verts.append([x[r,c], y[r,c], z[r,c]])
                    norms.append([nx[r,c], ny[r,c], nz[r,c]])
                    colors.append(m['colors'][r*cols + c])
            
            verts = np.array(verts)
            norms = np.array(norms)
            colors = np.array(colors)
            
            for r in range(rows - 1):
                for c in range(cols - 1):
                    p1 = r * cols + c
                    p2 = r * cols + (c + 1)
                    p3 = (r + 1) * cols + (c + 1)
                    p4 = (r + 1) * cols + c
                    
                    # Check for NaNs (hulling)
                    # If any vertex is NaN, don't create a face
                    if np.isnan(verts[p1][0]) or np.isnan(verts[p2][0]) or np.isnan(verts[p3][0]):
                        pass
                    else:
                        faces.append([p1, p2, p3])
                        
                    if np.isnan(verts[p1][0]) or np.isnan(verts[p3][0]) or np.isnan(verts[p4][0]):
                        pass
                    else:
                        faces.append([p1, p3, p4])
                    
            faces = np.array(faces)
            # Revert to auto-calculated normals by pyqtgraph to fix black rendering.
            # Manual injection of normals via _vertexNormals is unreliable.
            md = gl.MeshData(vertexes=verts, faces=faces, vertexColors=colors)
            
            mesh_item = gl.GLMeshItem(meshdata=md, smooth=True, shader='balloon', glOptions='opaque')
            self.view.addItem(mesh_item)

        self.el_work = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=8, cols=16, radius=0.35), color=(0, 1, 0, 1), shader='balloon')
        self.view.addItem(self.el_work)
        
        self.el_count = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=8, cols=16, radius=0.35), color=(1, 1, 0, 1), shader='balloon')
        self.view.addItem(self.el_count)
        
        self.line_loop = gl.GLLinePlotItem(pos=np.array([[0,0,0], [1,1,1]]), color=(255, 255, 255, 1.0), width=3)
        self.view.addItem(self.line_loop)

    def _update_visuals(self):
        self.el_work.resetTransform()
        self.el_work.translate(*self.pos_w_local)
        
        self.el_count.resetTransform()
        self.el_count.translate(*self.pos_c_local)
        
        p_w_lift = self.pos_w_local + np.array([0,0,0.2])
        p_c_lift = self.pos_c_local + np.array([0,0,0.2])
        self.line_loop.setData(pos=np.stack([p_w_lift, p_c_lift]))
        
        v_work = self.phantom.get_potential_at_point(self.pos_w_local)
        v_count = self.phantom.get_potential_at_point(self.pos_c_local)
        v_diff = v_work - v_count
        
        # Gain is relative to the "Reference Surface Potential" (1.5 mV)
        # If v_diff is 1.5 mV, gain is 1.0.
        # This gain is passed to BioEngine which scales the user's "Ref. ECG Amp".
        v_ref = 0.0015
        gain = v_diff / v_ref
        
        dist_cm = np.linalg.norm(self.pos_w_local - self.pos_c_local)
        
        # Display in mV for readability
        self.lbl_info.setText(f"Potentials: W={v_work*1000:.2f}mV | C={v_count*1000:.2f}mV | Diff={v_diff*1000:.2f}mV\nDist: {dist_cm:.1f}cm | Geometric Gain: {abs(gain):.2f}x")
        
        norm_w = [(self.pos_w_local[0] + 4)/8.0, (self.pos_w_local[1] + 2)/4.0] 
        norm_c = [(self.pos_c_local[0] + 4)/8.0, (self.pos_c_local[1] + 2)/4.0]
        self.geometry_changed.emit(gain, norm_w, norm_c)

    def get_positions(self):
        norm_w = [(self.pos_w_local[0] + 4)/8.0, (self.pos_w_local[1] + 2)/4.0] 
        norm_c = [(self.pos_c_local[0] + 4)/8.0, (self.pos_c_local[1] + 2)/4.0]
        return {"working": norm_w, "counter": norm_c}

    def set_tissue_resistivity(self, rho_ohm_cm):
        """
        Update phantom conductivity based on UI input.
        This affects potential field distribution and geometric gain.
        """
        self.phantom.set_resistivity(rho_ohm_cm)
        # Re-render to show updated potential map (colors might change brightness/scale)
        self._update_mesh_colors()
        # Do NOT emit signal back to UI to avoid recursion loop (UI -> recalc_rs -> set_rho -> signal -> UI)
        self._update_visuals(emit_signal=False)

    def _update_mesh_colors(self):
        # Regenerate mesh data with new potentials but keep geometry
        self.view.clear()
        self._init_scene()

    def _update_visuals(self, emit_signal=True):
        self.el_work.resetTransform()
        self.el_work.translate(*self.pos_w_local)
        
        self.el_count.resetTransform()
        self.el_count.translate(*self.pos_c_local)
        
        p_w_lift = self.pos_w_local + np.array([0,0,0.2])
        p_c_lift = self.pos_c_local + np.array([0,0,0.2])
        self.line_loop.setData(pos=np.stack([p_w_lift, p_c_lift]))
        
        v_work = self.phantom.get_potential_at_point(self.pos_w_local)
        v_count = self.phantom.get_potential_at_point(self.pos_c_local)
        v_diff = v_work - v_count
        
        # Gain is relative to the "Reference Surface Potential" (1.5 mV)
        v_ref = 0.0015
        gain = v_diff / v_ref
        
        dist_cm = np.linalg.norm(self.pos_w_local - self.pos_c_local)
        
        # Display in mV for readability
        self.lbl_info.setText(f"Potentials: W={v_work*1000:.2f}mV | C={v_count*1000:.2f}mV | Diff={v_diff*1000:.2f}mV\nDist: {dist_cm:.1f}cm | Geometric Gain: {abs(gain):.2f}x")
        
        if emit_signal:
            norm_w = [(self.pos_w_local[0] + 4)/8.0, (self.pos_w_local[1] + 2)/4.0] 
            norm_c = [(self.pos_c_local[0] + 4)/8.0, (self.pos_c_local[1] + 2)/4.0]
            self.geometry_changed.emit(gain, norm_w, norm_c)
    
    def _get_ray(self, x, y):
        w = self.view.width()
        h = self.view.height()
        
        cam_pos = self.view.cameraPosition() 
        center = self.view.opts['center'] 
        fov = self.view.opts['fov']
        
        fwd = (center - cam_pos).normalized()
        up_global = QtGui.QVector3D(0,0,1)
        right = QtGui.QVector3D.crossProduct(fwd, up_global).normalized()
        up = QtGui.QVector3D.crossProduct(right, fwd).normalized()
        
        ndc_x = (2.0 * x) / w - 1.0
        ndc_y = 1.0 - (2.0 * y) / h
        
        aspect = w / h
        tan_half = np.tan(np.radians(fov) / 2.0)
        
        ray_dir = fwd + right * (ndc_x * aspect * tan_half) + up * (ndc_y * tan_half)
        ray_dir.normalize()
        
        return np.array([cam_pos.x(), cam_pos.y(), cam_pos.z()]), np.array([ray_dir.x(), ray_dir.y(), ray_dir.z()])

    def _dist_ray_point(self, ray_origin, ray_dir, point):
        po = point - ray_origin
        cross = np.cross(po, ray_dir)
        return np.linalg.norm(cross)

    # --- EVENT HANDLING ---

    def _try_start_drag(self, pos):
        try:
            ray_o, ray_d = self._get_ray(pos.x(), pos.y())
            # Increased threshold for easier grabbing
            radius_threshold = 1.5 
            
            dist_w = self._dist_ray_point(ray_o, ray_d, self.pos_w_local)
            dist_c = self._dist_ray_point(ray_o, ray_d, self.pos_c_local)
            
            # Prioritize closer electrode if both are in range
            if dist_w < radius_threshold and dist_w < dist_c:
                self.dragging_electrode = 'work'
                return True
            elif dist_c < radius_threshold:
                self.dragging_electrode = 'count'
                return True
        except:
            pass
        return False

    def _handle_drag(self, pos):
        # Ray Cast to PHANTOM GEOMETRY
        ray_o, ray_d = self._get_ray(pos.x(), pos.y())
        
        # Intersect ray with all ellipsoids
        hit_point = self.phantom.intersect_ray(ray_o, ray_d)
        
        if hit_point is not None:
            # Update
            if self.dragging_electrode == 'work':
                self.pos_w_local = hit_point
            elif self.dragging_electrode == 'count':
                self.pos_c_local = hit_point
                
            self._update_visuals()

    def eventFilter(self, source, event):
        if source == self.view:
            if event.type() == QtCore.QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    if self._try_start_drag(event.pos()):
                        return True 
                        
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                if self.dragging_electrode:
                    self._handle_drag(event.pos())
                    return True 
                    
            elif event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                if self.dragging_electrode:
                    self.dragging_electrode = None
                    return True
                    
        return super().eventFilter(source, event)
