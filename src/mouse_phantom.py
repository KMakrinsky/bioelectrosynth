import numpy as np

class MousePhantom:
    """
    Physiological 3D model of a mouse (High Fidelity).
    Implements Volume Conductor Physics for potential calculation.
    """
    def __init__(self, rho_ohm_cm=100.0):
        # --- PHYSICS CONSTANTS ---
        # rho (Ohm*cm) -> sigma (S/m)
        # 1 Ohm*cm = 0.01 Ohm*m
        # sigma = 1 / (rho * 0.01) = 100 / rho
        self.set_resistivity(rho_ohm_cm)
        
        self.SCALE = 0.01     # 1 unit = 1 cm = 0.01 m
        
        # --- HEART DIPOLE ---
        self.heart_pos = np.array([1.2, 0.0, -0.2]) 
        
        # Dipole Orientation: Down (-Z), Left (+Y), slightly Back (-X)
        direction = np.array([-0.5, 1.0, -0.5]) 
        direction = direction / np.linalg.norm(direction)
        
        # Dipole Magnitude Calculation:
        # We define P such that at Standard Conductivity (rho=100 Ohm*cm -> sigma=1.0),
        # the potential at 1cm is 1.5 mV.
        # If conductivity changes, the potential should change physically (V ~ 1/sigma).
        # So we FIX the dipole moment P (source strength), and V varies with sigma.
        
        sigma_std = 1.0 # S/m (corresponding to 100 Ohm*cm)
        target_V = 0.0015
        r_ref = 0.01 
        
        # P = V * 4 * pi * sigma * r^2
        P_mag = target_V * 4 * np.pi * sigma_std * (r_ref**2)
        
        self.dipole_moment = direction * P_mag
        
        # Geometry Parts (V2 Logic)
        self.parts = [
            # 1. Main Body (Rear/Rump)
            {'c': np.array([-1.5, 0, 0.2]), 'r': np.array([2.0, 1.8, 1.7]), 'rot': -15, 'type': 'body'},
            # 2. Chest (Front)
            {'c': np.array([1.0, 0, 0.4]),  'r': np.array([2.2, 1.7, 1.6]), 'rot': 10,  'type': 'body'},
            # 3. Head
            {'c': np.array([3.5, 0, -0.3]), 'r': np.array([1.4, 1.1, 1.0]), 'rot': 20,  'type': 'head'},
            # 4. Ears
            {'c': np.array([3.2, 0.9, 0.8]), 'r': np.array([0.2, 0.7, 0.7]), 'rot': 10, 'type': 'ear', 'attach': np.array([3.0, 0.5, 0.5])},
            {'c': np.array([3.2, -0.9, 0.8]), 'r': np.array([0.2, 0.7, 0.7]), 'rot': 10, 'type': 'ear', 'attach': np.array([3.0, -0.5, 0.5])},
            # 5. Thighs (Rear Legs)
            {'c': np.array([-1.8, 1.4, -0.5]), 'r': np.array([0.9, 0.6, 1.0]), 'rot': -20, 'type': 'leg'},
            {'c': np.array([-1.8, -1.4, -0.5]), 'r': np.array([0.9, 0.6, 1.0]), 'rot': -20, 'type': 'leg'},
            # 6. Forelimbs (Front Legs)
            {'c': np.array([1.5, 1.2, -0.6]), 'r': np.array([0.7, 0.5, 0.8]), 'rot': 15, 'type': 'leg'},
            {'c': np.array([1.5, -1.2, -0.6]), 'r': np.array([0.7, 0.5, 0.8]), 'rot': 15, 'type': 'leg'},
        ]

    def set_resistivity(self, rho_ohm_cm):
        # rho in Ohm*cm -> sigma in S/m
        # 1 Ohm*cm = 0.01 Ohm*m
        # sigma = 1 / (rho * 0.01)
        if rho_ohm_cm < 1e-3: rho_ohm_cm = 1e-3
        self.SIGMA = 1.0 / (rho_ohm_cm * 0.01)

    def intersect_ray(self, ray_origin, ray_dir):
        """
        Finds the closest intersection of a ray with any of the ellipsoids.
        Returns the intersection point (np.array) or None.
        """
        closest_t = float('inf')
        closest_point = None
        
        for part in self.parts:
            # Transform Ray to Local Space
            ang = np.radians(part['rot'])
            cos_a = np.cos(ang)
            sin_a = np.sin(ang)
            
            def to_local(v):
                x = v[0] * cos_a - v[2] * sin_a
                y = v[1]
                z = v[0] * sin_a + v[2] * cos_a
                return np.array([x, y, z])
                
            O_loc = to_local(ray_origin - part['c'])
            D_loc = to_local(ray_dir)
            
            radii = part['r']
            O_s = O_loc / radii
            D_s = D_loc / radii
            
            a = np.dot(D_s, D_s)
            b = 2.0 * np.dot(O_s, D_s)
            c = np.dot(O_s, O_s) - 1.0
            
            delta = b*b - 4*a*c
            
            if delta >= 0:
                sqrt_delta = np.sqrt(delta)
                t1 = (-b - sqrt_delta) / (2*a)
                t2 = (-b + sqrt_delta) / (2*a)
                
                candidates = []
                if t1 > 0: candidates.append(t1)
                if t2 > 0: candidates.append(t2)
                
                if candidates:
                    t_s = min(candidates)
                    if t_s < closest_t:
                        closest_t = t_s
                        closest_point = ray_origin + t_s * ray_dir

        return closest_point

    def snap_to_surface(self, x: float, y: float) -> np.ndarray:
        """
        Legacy Z-search. Kept for initialization.
        """
        max_z = -float('inf')
        for part in self.parts:
            z = self._get_ellipsoid_z_surface(x, y, part)
            if z > max_z:
                max_z = z
        if max_z == -float('inf'):
            return np.array([x, y, 0.0])
        return np.array([x, y, max_z])

    def _get_ellipsoid_z_surface(self, x_glob, y_glob, part):
        p_rel = np.array([x_glob, y_glob, 0]) - part['c']
        z_guess = part['c'][2] + part['r'][2]
        for _ in range(5):
            vec = np.array([x_glob, y_glob, z_guess]) - part['c']
            ang = np.radians(-part['rot'])
            x_loc = vec[0]*np.cos(ang) - vec[2]*np.sin(ang)
            y_loc = vec[1]
            z_loc = vec[0]*np.sin(ang) + vec[2]*np.cos(ang)
            term = (x_loc / part['r'][0])**2 + (y_loc / part['r'][1])**2
            if term > 1.0: return -float('inf')
            z_loc_surf = part['r'][2] * np.sqrt(1.0 - term)
            ang_fwd = np.radians(part['rot'])
            z_glob = (x_loc * np.sin(ang_fwd) + z_loc_surf * np.cos(ang_fwd)) + part['c'][2]
            z_guess = z_glob
        return z_guess

    def get_potential_at_point(self, point: np.ndarray) -> float:
        """
        Calculates physical potential (Volts) using Volume Conductor Theory.
        Phi = (P . r) / (4 * pi * sigma * r^3)
        """
        # Vector from dipole to point
        r_vec_cm = point - self.heart_pos
        
        # Convert to meters
        r_vec_m = r_vec_cm * self.SCALE
        r_len_m = np.linalg.norm(r_vec_m)
        
        # Singularity protection (inside heart) - clamp to 1mm
        if r_len_m < 0.001: r_len_m = 0.001
        
        # Dot product
        dot_prod = np.dot(self.dipole_moment, r_vec_m)
        
        # Formula
        k = 1.0 / (4.0 * np.pi * self.SIGMA)
        phi = k * dot_prod / (r_len_m**3)
        
        return phi

    def is_point_inside_part(self, p_glob, part):
        vec = p_glob - part['c']
        ang = np.radians(-part['rot'])
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)
        x_loc = vec[0] * cos_a - vec[2] * sin_a
        y_loc = vec[1]
        z_loc = vec[0] * sin_a + vec[2] * cos_a
        metric = (x_loc / part['r'][0])**2 + (y_loc / part['r'][1])**2 + (z_loc / part['r'][2])**2
        return metric < 0.90

    def generate_mesh_data(self, hulling=True, u_res=30, v_res=20):
        meshes = []
        for part_idx, part in enumerate(self.parts):
            u = np.linspace(0, 2 * np.pi, u_res)
            v = np.linspace(0, np.pi, v_res)
            
            # Local coordinates
            x0 = part['r'][0] * np.outer(np.cos(u), np.sin(v))
            y0 = part['r'][1] * np.outer(np.sin(u), np.sin(v))
            z0 = part['r'][2] * np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Normals
            nx0 = x0 / (part['r'][0]**2)
            ny0 = y0 / (part['r'][1]**2)
            nz0 = z0 / (part['r'][2]**2)
            n_mag = np.sqrt(nx0**2 + ny0**2 + nz0**2)
            nx0 /= n_mag
            ny0 /= n_mag
            nz0 /= n_mag
            
            theta = np.radians(part['rot'])
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Rotate Geometry
            x_rot = x0 * cos_t + z0 * sin_t
            z_rot = -x0 * sin_t + z0 * cos_t
            x_fin = x_rot + part['c'][0]
            y_fin = y0 + part['c'][1]
            z_fin = z_rot + part['c'][2]
            
            # Flatten
            pts = np.stack([x_fin.flatten(), y_fin.flatten(), z_fin.flatten()], axis=1)
            colors = np.zeros((len(pts), 4))
            
            attach_p = part.get('attach', None)
            
            if hulling:
                for i, p in enumerate(pts):
                    is_hidden = False
                    for other_idx, other_part in enumerate(self.parts):
                        if other_idx == part_idx: continue
                        if self.is_point_inside_part(p, other_part):
                            is_hidden = True
                            break
                    if is_hidden:
                        pts[i] = [np.nan, np.nan, np.nan]
                        continue
            
            # Coloring by Potential
            for i, p in enumerate(pts):
                if np.isnan(p[0]): continue 

                if attach_p is not None:
                    v_attach = self.get_potential_at_point(attach_p)
                    v_local = self.get_potential_at_point(p)
                    pot = 0.8 * v_attach + 0.2 * v_local
                else:
                    pot = self.get_potential_at_point(p)
                
                # Auto-scale for visualization. 
                # Our typical range is +/- 1.5 mV (0.0015 V) on surface, 
                # but can be +/- 20 mV near heart.
                # Let's clip visual range to +/- 2 mV for high sensitivity on surface.
                limit = 0.002
                norm = np.clip((pot + limit)/(2*limit), 0, 1)
                
                if norm < 0.5:
                    t = norm * 2.0
                    r, g, b = t, t, 1.0 # Blue to White
                else:
                    t = (norm - 0.5) * 2.0
                    r, g, b = 1.0, 1.0 - t, 1.0 - t # White to Red
                colors[i] = [r, g, b, 1.0] 
            
            x_fin_masked = pts[:, 0].reshape(x_fin.shape)
            y_fin_masked = pts[:, 1].reshape(y_fin.shape)
            z_fin_masked = pts[:, 2].reshape(z_fin.shape)
            
            # Reconstruct Normals for Widget (masking not strictly needed for normals if coords are NaN, but safer)
            nx_masked = np.zeros_like(x_fin_masked)
            ny_masked = np.zeros_like(y_fin_masked)
            nz_masked = np.zeros_like(z_fin_masked)
            
            # Since we rotated geometry, we must rotate normals too
            nx_rot = nx0 * cos_t + nz0 * sin_t
            ny_rot = ny0
            nz_rot = -nx0 * sin_t + nz0 * cos_t
            
            # Apply same flattening/masking logic if needed, or just reshape if no hulling changed indices
            # Hulling sets pts to NaN. We should set normals to NaN too for consistency?
            # Actually, we didn't store modified normals in pts loop.
            # Let's just re-calculate rotated normals shape.
            
            meshes.append({
                'x': x_fin_masked, 'y': y_fin_masked, 'z': z_fin_masked, 
                'nx': nx_rot, 'ny': ny_rot, 'nz': nz_rot,
                'colors': colors
            })
        return meshes
