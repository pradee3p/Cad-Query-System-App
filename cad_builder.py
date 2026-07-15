import os
import json
import math
import sys
import base64
import atexit

# Global state for parameters and scenes
_params = {}
_scene_shapes = []

# Load parameters from JSON file (set by Express server)
_params_file = os.environ.get("CAD_PARAMS_FILE")
if _params_file and os.path.exists(_params_file):
    try:
        with open(_params_file, "r") as f:
            _params = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Error loading params file: {str(e)}\n")

def get_param(name, default_value=0.0):
    """Gets a parameter value. Returns default if not found."""
    val = _params.get(name, default_value)
    # Ensure float or int
    try:
        if isinstance(default_value, int):
            return int(val)
        return float(val)
    except Exception:
        return default_value

class Mesh:
    def __init__(self, vertices=None, faces=None):
        self.vertices = vertices if vertices is not None else []
        self.faces = faces if faces is not None else []

    def copy(self):
        return Mesh([v[:] for v in self.vertices], [f[:] for f in self.faces])

    def translate(self, x, y, z):
        new_mesh = self.copy()
        for i in range(len(new_mesh.vertices)):
            new_mesh.vertices[i][0] += x
            new_mesh.vertices[i][1] += y
            new_mesh.vertices[i][2] += z
        return new_mesh

    def rotate_x(self, degrees):
        rad = math.radians(degrees)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_mesh = self.copy()
        for i in range(len(new_mesh.vertices)):
            _, y, z = new_mesh.vertices[i]
            new_mesh.vertices[i][1] = y * cos_a - z * sin_a
            new_mesh.vertices[i][2] = y * sin_a + z * cos_a
        return new_mesh

    def rotate_y(self, degrees):
        rad = math.radians(degrees)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_mesh = self.copy()
        for i in range(len(new_mesh.vertices)):
            x, _, z = new_mesh.vertices[i]
            new_mesh.vertices[i][0] = x * cos_a + z * sin_a
            new_mesh.vertices[i][2] = -x * sin_a + z * cos_a
        return new_mesh

    def rotate_z(self, degrees):
        rad = math.radians(degrees)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        new_mesh = self.copy()
        for i in range(len(new_mesh.vertices)):
            x, y, _ = new_mesh.vertices[i]
            new_mesh.vertices[i][0] = x * cos_a - y * sin_a
            new_mesh.vertices[i][1] = x * sin_a + y * cos_a
        return new_mesh

    def __add__(self, other):
        """Union of two meshes (combines vertex/face arrays)"""
        if not isinstance(other, Mesh):
            return self
        
        new_vertices = [v[:] for v in self.vertices]
        new_faces = [f[:] for f in self.faces]
        v_offset = len(new_vertices)
        
        for v in other.vertices:
            new_vertices.append(v[:])
        for f in other.faces:
            new_faces.append([f[0] + v_offset, f[1] + v_offset, f[2] + v_offset])
            
        return Mesh(new_vertices, new_faces)

    def __sub__(self, other):
        """Difference (simplified CSG subtraction via bounding check/distance)"""
        # For simple and robust pure python modeling, difference can be performed 
        # by discarding triangles of Mesh A that fall fully inside Mesh B,
        # or we can use a voxel-based approach. Since we want standard shapes to look correct:
        # We will write a smart voxel-based subtraction OR discard faces close to/inside the subtracting shape.
        # Wait, if trimesh is installed, we can delegate to trimesh! Let's check:
        try:
            import trimesh
            import numpy as np
            # Convert self and other to trimesh
            t1 = trimesh.Trimesh(vertices=np.array(self.vertices), faces=np.array(self.faces))
            t2 = trimesh.Trimesh(vertices=np.array(other.vertices), faces=np.array(other.faces))
            # Perform subtraction using trimesh boolean
            diff = t1.difference(t2, engine='scad' if hasattr(t1, 'scad') else None)
            return Mesh(diff.vertices.tolist(), diff.faces.tolist())
        except Exception:
            # Fallback pure python subtraction (discards faces of self inside other)
            # To determine if a vertex is inside other, we check relative distance from its center
            return self._fallback_difference(other)

    def __and__(self, other):
        """Intersection of two meshes"""
        try:
            import trimesh
            import numpy as np
            t1 = trimesh.Trimesh(vertices=np.array(self.vertices), faces=np.array(self.faces))
            t2 = trimesh.Trimesh(vertices=np.array(other.vertices), faces=np.array(other.faces))
            inter = t1.intersection(t2)
            return Mesh(inter.vertices.tolist(), inter.faces.tolist())
        except Exception:
            return self

    def _fallback_difference(self, other):
        """Discard vertices of self that are inside the bounding sphere/box of other."""
        # Calculate bounding sphere of 'other'
        if not other.vertices:
            return self.copy()
            
        # Basic centroid of other
        cx = sum(v[0] for v in other.vertices) / len(other.vertices)
        cy = sum(v[1] for v in other.vertices) / len(other.vertices)
        cz = sum(v[2] for v in other.vertices) / len(other.vertices)
        
        # Approximate max radius
        r_sq = 0
        for v in other.vertices:
            d = (v[0] - cx)**2 + (v[1] - cy)**2 + (v[2] - cz)**2
            if d > r_sq:
                r_sq = d
        r_other = math.sqrt(r_sq)

        # Discard self vertices that are inside other
        # Note: to avoid breaking faces entirely, we can reconstruct or just filter
        # For visualization, we keep all faces where at least one vertex is outside other
        new_vertices = []
        old_to_new_index = {}
        
        for idx, v in enumerate(self.vertices):
            dist_sq = (v[0] - cx)**2 + (v[1] - cy)**2 + (v[2] - cz)**2
            # Check if inside
            is_inside = dist_sq < (r_other * 0.9)**2 # slightly smaller to avoid edge clipping
            if not is_inside:
                old_to_new_index[idx] = len(new_vertices)
                new_vertices.append(v)

        new_faces = []
        for f in self.faces:
            if f[0] in old_to_new_index and f[1] in old_to_new_index and f[2] in old_to_new_index:
                new_faces.append([
                    old_to_new_index[f[0]],
                    old_to_new_index[f[1]],
                    old_to_new_index[f[2]]
                ])
                
        return Mesh(new_vertices, new_faces)

    def compute_volume(self):
        """Calculates volume of the mesh (using standard tetrahedron method)"""
        vol = 0.0
        for f in self.faces:
            if len(f) < 3:
                continue
            v0 = self.vertices[f[0]]
            v1 = self.vertices[f[1]]
            v2 = self.vertices[f[2]]
            
            # Signed volume of tetrahedron
            v321 = v2[0] * v1[1] * v0[2]
            v231 = v1[0] * v2[1] * v0[2]
            v312 = v2[0] * v0[1] * v1[2]
            v132 = v0[0] * v2[1] * v1[2]
            v213 = v1[0] * v0[1] * v2[2]
            v123 = v0[0] * v1[1] * v2[2]
            
            vol += (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123)
            
        return abs(vol)

    def compute_surface_area(self):
        """Calculates the surface area of the mesh"""
        area = 0.0
        for f in self.faces:
            v0 = self.vertices[f[0]]
            v1 = self.vertices[f[1]]
            v2 = self.vertices[f[2]]
            
            # Vector cross product
            ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
            bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
            
            cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
            face_area = 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)
            area += face_area
        return area

    def get_bounding_box(self):
        """Returns bounds as (min_coords, max_coords)"""
        if not self.vertices:
            return [0, 0, 0], [0, 0, 0]
            
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        
        return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


# --- Procedural Primitives ---

def Box(width, height, depth):
    w, h, d = width / 2.0, height / 2.0, depth / 2.0
    vertices = [
        [-w, -h, -d], [w, -h, -d], [w, h, -d], [-w, h, -d], # Bottom
        [-w, -h, d],  [w, -h, d],  [w, h, d],  [-w, h, d]   # Top
    ]
    # Triangles with correct orientation (counter-clockwise looking from outside)
    faces = [
        [0, 2, 1], [0, 3, 2], # Bottom
        [4, 5, 6], [4, 6, 7], # Top
        [0, 1, 5], [0, 5, 4], # Front
        [1, 2, 6], [1, 6, 5], # Right
        [2, 3, 7], [2, 7, 6], # Back
        [3, 0, 4], [3, 4, 7]  # Left
    ]
    return Mesh(vertices, faces)

def Cylinder(radius, height, sections=32):
    sections = max(3, int(sections))
    vertices = []
    faces = []
    h2 = height / 2.0

    # Create bottom and top circle vertices
    for i in range(sections):
        angle = 2.0 * math.pi * i / sections
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        # Bottom ring
        vertices.append([x, y, -h2])
        # Top ring
        vertices.append([x, y, h2])

    # Add centers of caps
    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, -h2])
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, h2])

    # Side faces
    for i in range(sections):
        next_i = (i + 1) % sections
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * next_i
        t1 = 2 * next_i + 1

        # Triangle 1
        faces.append([b0, b1, t0])
        # Triangle 2
        faces.append([t0, b1, t1])

    # Cap faces
    for i in range(sections):
        next_i = (i + 1) % sections
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * next_i
        t1 = 2 * next_i + 1

        # Bottom cap (clock-wise seen from bottom, CCW from outside)
        faces.append([b0, bottom_center_idx, b1])
        # Top cap (CCW seen from top, CCW from outside)
        faces.append([t0, t1, top_center_idx])

    return Mesh(vertices, faces)

def Sphere(radius, sections=32):
    sections = max(4, int(sections))
    # UV Sphere generation
    vertices = []
    faces = []
    
    # We will use subdivisions: latitude (rings) and longitude (segments)
    rings = sections // 2
    segments = sections

    for r in range(rings + 1):
        phi = math.pi * r / rings # Latitude angle
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        
        for s in range(segments):
            theta = 2.0 * math.pi * s / segments # Longitude angle
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            
            x = radius * sin_phi * cos_theta
            y = radius * sin_phi * sin_theta
            z = radius * cos_phi
            vertices.append([x, y, z])

    # Connect faces
    for r in range(rings):
        for s in range(segments):
            s_next = (s + 1) % segments
            
            # Index positions
            i0 = r * segments + s
            i1 = r * segments + s_next
            i2 = (r + 1) * segments + s
            i3 = (r + 1) * segments + s_next
            
            # Triangles
            if r != 0:
                faces.append([i0, i1, i2])
            if r != rings - 1:
                faces.append([i1, i3, i2])
                
    return Mesh(vertices, faces)

def Cone(radius1, radius2, height, sections=32):
    sections = max(3, int(sections))
    vertices = []
    faces = []
    h2 = height / 2.0

    # Bottom and top circles
    for i in range(sections):
        angle = 2.0 * math.pi * i / sections
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        vertices.append([radius1 * cos_a, radius1 * sin_a, -h2]) # Bottom ring
        vertices.append([radius2 * cos_a, radius2 * sin_a, h2])  # Top ring

    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, -h2])
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, h2])

    # Sides
    for i in range(sections):
        next_i = (i + 1) % sections
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * next_i
        t1 = 2 * next_i + 1

        faces.append([b0, b1, t0])
        faces.append([t0, b1, t1])

    # Caps
    for i in range(sections):
        next_i = (i + 1) % sections
        b0 = 2 * i
        t0 = 2 * i + 1
        b1 = 2 * next_i
        t1 = 2 * next_i + 1

        faces.append([b0, bottom_center_idx, b1])
        faces.append([t0, t1, top_center_idx])

    return Mesh(vertices, faces)

def Torus(major_radius, minor_radius, sections=32):
    sections = max(4, int(sections))
    vertices = []
    faces = []
    
    # Grid of points: theta (around torus loop), phi (around tube circle)
    num_u = sections
    num_v = sections // 2 + 2

    for u in range(num_u):
        theta = 2.0 * math.pi * u / num_u
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        for v in range(num_v):
            phi = 2.0 * math.pi * v / num_v
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            
            # Position
            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = (major_radius + minor_radius * cos_phi) * sin_theta
            z = minor_radius * sin_phi
            vertices.append([x, y, z])

    # Connect faces
    for u in range(num_u):
        u_next = (u + 1) % num_u
        for v in range(num_v):
            v_next = (v + 1) % num_v
            
            i0 = u * num_v + v
            i1 = u_next * num_v + v
            i2 = u * num_v + v_next
            i3 = u_next * num_v + v_next
            
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])
            
    return Mesh(vertices, faces)

def Hexagon(width, height):
    # Hexagon width flat-to-flat -> circumradius = width / cos(30 deg) = width / 0.866025
    r = (width / 2.0) / math.cos(math.radians(30))
    return Cylinder(radius=r, height=height, sections=6)

def ThreadedShaft(diameter, length, pitch=1.5):
    # A simplified, beautiful threaded shaft mesh.
    # It creates a cylinder but modulates the radius sinusoidally along the height!
    sections = 32
    vertices = []
    faces = []
    h2 = length / 2.0
    r_core = diameter / 2.0
    
    # We will subdivide along the length to show the beautiful threads
    subdivisions = int(max(10, length / (pitch / 4.0)))
    
    for ring in range(subdivisions + 1):
        z = -h2 + (length * ring / subdivisions)
        # Calculate pitch phase: helical angle
        phase = 2.0 * math.pi * (z / pitch)
        
        for s in range(sections):
            angle = 2.0 * math.pi * s / sections
            # Radius varies sinusoidally based on angle and height to create helical thread effect!
            # r = core_r + depth * sin(theta - z_phase)
            depth = diameter * 0.08 # thread depth
            r = r_core + depth * math.sin(angle - phase)
            
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            vertices.append([x, y, z])

    # Connect side faces
    for ring in range(subdivisions):
        for s in range(sections):
            s_next = (s + 1) % sections
            
            i0 = ring * sections + s
            i1 = ring * sections + s_next
            i2 = (ring + 1) * sections + s
            i3 = (ring + 1) * sections + s_next
            
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    # Add caps
    bottom_center_idx = len(vertices)
    vertices.append([0.0, 0.0, -h2])
    top_center_idx = len(vertices)
    vertices.append([0.0, 0.0, h2])

    # Connect caps
    for s in range(sections):
        s_next = (s + 1) % sections
        b0 = s
        b1 = s_next
        t0 = subdivisions * sections + s
        t1 = subdivisions * sections + s_next

        # Bottom cap
        faces.append([b0, bottom_center_idx, b1])
        # Top cap
        faces.append([t0, t1, top_center_idx])

    return Mesh(vertices, faces)


# --- Standard Assemblies / Helper CAD Functions ---

def create_hex_bolt(diameter, length, head_width, head_height):
    """Creates a hex bolt assembly"""
    # 1. Threaded shaft (centered at origin, extending from -length/2 to length/2)
    shaft = ThreadedShaft(diameter, length).translate(0, 0, length / 2.0)
    # 2. Hex head (extends from -head_height/2 to head_height/2)
    # Put hex head directly below shaft (translate it down)
    head = Hexagon(head_width, head_height).translate(0, 0, -head_height / 2.0)
    
    # Merge and center them so the plane where head meets shaft is at Z = 0
    return head + shaft

def create_nut(diameter, thickness, outer_width, hole_tolerance=0.4):
    """Creates a hex nut with a threaded hole"""
    # Create the hex outer shell
    nut_body = Hexagon(outer_width, thickness)
    # Create a threaded shaft representing the bolt hole (with a tiny clearance tolerance)
    hole_diameter = diameter + hole_tolerance
    # We will subtract the threaded shaft from the hexagon body!
    # Wait, simple subtraction falls back nicely, or we can use trimesh if installed.
    hole = ThreadedShaft(hole_diameter, thickness * 1.2) # slightly taller to ensure clean cut
    
    return nut_body - hole

def create_bracket(width, height, depth, hole_diameter, thickness=3.0):
    """Creates an L-bracket with a mounting hole on each flange"""
    # Horizontal flange
    horiz = Box(width, thickness, depth).translate(0, thickness/2.0, 0)
    # Vertical flange
    vert = Box(thickness, height, depth).translate(-width/2.0 + thickness/2.0, height/2.0 + thickness/2.0, 0)
    
    bracket_body = horiz + vert
    
    # Create holes
    hole1 = Cylinder(hole_diameter / 2.0, thickness * 2.0).rotate_x(90).translate(0, thickness/2.0, 0)
    hole2 = Cylinder(hole_diameter / 2.0, thickness * 2.0).rotate_y(90).translate(-width/2.0 + thickness/2.0, height/2.0 + thickness/2.0, 0)
    
    return (bracket_body - hole1) - hole2

def create_gear(teeth, thickness, outer_radius, inner_radius, bore_diameter):
    """Creates a spur gear with custom teeth and bore shaft"""
    # Base core cylinder
    core = Cylinder(inner_radius, thickness)
    
    # Add teeth
    # Each tooth is represented by a small box, rotated and translated around the perimeter
    mesh = core
    tooth_angle = 360.0 / teeth
    tooth_width = (2 * math.pi * outer_radius) / (teeth * 2)
    tooth_length = outer_radius - inner_radius
    
    for i in range(teeth):
        angle = tooth_angle * i
        # Tooth box centered on outer ring
        tooth = Box(tooth_length, tooth_width, thickness).translate(inner_radius + tooth_length/2.0, 0, 0).rotate_z(angle)
        mesh = mesh + tooth
        
    # Subtract center bore shaft
    bore = Cylinder(bore_diameter / 2.0, thickness * 1.2)
    return mesh - bore


# --- Scene Management & Exports ---

def add_shape(mesh):
    """Adds a mesh to the global scene to be exported."""
    if isinstance(mesh, Mesh):
        _scene_shapes.append(mesh)

def export_scene_data():
    """Generates the combined STL mesh and metadata JSON, printing it to stdout."""
    if not _scene_shapes:
        sys.stderr.write("No shapes added to the scene. Creating a default box.\n")
        add_shape(Box(10, 10, 10))

    # Combine all shapes into a single master mesh
    master_mesh = Mesh()
    for shape in _scene_shapes:
        master_mesh = master_mesh + shape

    # Compute properties
    volume = master_mesh.compute_volume()
    surface_area = master_mesh.compute_surface_area()
    bbox_min, bbox_max = master_mesh.get_bounding_box()
    
    # Generate STL content
    stl_str = generate_ascii_stl(master_mesh)
    stl_base64 = base64.b64encode(stl_str.encode("utf-8")).decode("utf-8")

    # Final payload
    payload = {
        "stl": stl_base64,
        "metadata": {
            "volume": round(volume, 2),
            "surface_area": round(surface_area, 2),
            "bbox_min": [round(c, 2) for c in bbox_min],
            "bbox_max": [round(c, 2) for c in bbox_max],
            "vertices_count": len(master_mesh.vertices),
            "faces_count": len(master_mesh.faces)
        }
    }

    # Print results inside delimiters for Express server parsing
    print("===CAD_OUTPUT_START===")
    print(json.dumps(payload))
    print("===CAD_OUTPUT_END===")

def generate_ascii_stl(mesh, name="CAD_Query_Model"):
    """Helper to generate standard ASCII STL string from a Mesh object"""
    lines = [f"solid {name}"]
    
    for f in mesh.faces:
        if len(f) < 3:
            continue
        v0 = mesh.vertices[f[0]]
        v1 = mesh.vertices[f[1]]
        v2 = mesh.vertices[f[2]]
        
        # Calculate normal vector
        ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
        cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
        length = math.sqrt(cx*cx + cy*cy + cz*cz)
        if length > 0:
            nx, ny, nz = cx / length, cy / length, cz / length
        else:
            nx, ny, nz = 0.0, 0.0, 0.0
            
        lines.append(f"  facet normal {nx:.6f} {ny:.6f} {nz:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}")
        lines.append(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}")
        lines.append(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}")
        lines.append("    endloop")
        lines.append("  endfacet")
        
    lines.append(f"endsolid {name}")
    return "\n".join(lines)


# Automatically export the scene at script completion
@atexit.register
def _auto_export():
    # Only export if add_shape has been called
    if _scene_shapes:
        export_scene_data()
