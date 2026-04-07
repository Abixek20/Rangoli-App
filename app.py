"""
Rangoli Pattern AI - Flask Backend
Analyzes and generates traditional Rangoli patterns using
geometric symmetry, graph representations, and algorithmic design.
"""

import os
import io
import json
import base64
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from scipy import ndimage
from scipy.spatial import Delaunay
import networkx as nx
from sklearn.cluster import KMeans

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'rangoli-dev-secret-change-in-prod')
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16 MB
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────────────────────────────────────────
# 1. GEOMETRIC PATTERN ENGINE (Matrix-based)
# ─────────────────────────────────────────────

class RangoliPatternEngine:
    """Generates Rangoli patterns using matrix transformations and geometric principles."""

    def __init__(self, size=800):
        self.size = size
        self.center = (size // 2, size // 2)
        self.pattern_matrix = np.zeros((size, size, 4), dtype=np.uint8)

    def _rotation_matrix(self, angle_deg):
        """Create a 2D rotation matrix."""
        theta = math.radians(angle_deg)
        return np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ])

    def _reflect_point(self, point, axis_angle):
        """Reflect a point across an axis at given angle through center."""
        cx, cy = self.center
        px, py = point[0] - cx, point[1] - cy
        theta = 2 * math.radians(axis_angle)
        rx = px * math.cos(theta) + py * math.sin(theta)
        ry = px * math.sin(theta) - py * math.cos(theta)
        return (rx + cx, ry + cy)

    def generate_dot_grid(self, rows, cols, spacing):
        """Generate the traditional dot grid (pulli kolam style)."""
        dots = []
        cx, cy = self.center
        for i in range(rows):
            for j in range(cols):
                x = cx + (j - cols // 2) * spacing
                y = cy + (i - rows // 2) * spacing
                dots.append((x, y))
        return dots

    def generate_radial_points(self, num_folds, layers, base_radius):
        """Generate points with n-fold radial symmetry."""
        points = []
        for layer in range(1, layers + 1):
            radius = base_radius * layer / layers
            num_points = num_folds * layer
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                x = self.center[0] + radius * math.cos(angle)
                y = self.center[1] + radius * math.sin(angle)
                points.append((x, y))
        return points

    def generate_petal_curve(self, cx, cy, radius, num_petals, petal_width, rotation=0):
        """Generate petal/lotus curves using parametric equations."""
        curves = []
        for p in range(num_petals):
            base_angle = 2 * math.pi * p / num_petals + math.radians(rotation)
            curve_points = []
            for t in np.linspace(-petal_width, petal_width, 40):
                r = radius * math.cos(t * num_petals / 2)
                if r < 0:
                    continue
                angle = base_angle + t
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                curve_points.append((x, y))
            if curve_points:
                curves.append(curve_points)
        return curves

    def generate_spiral(self, cx, cy, max_radius, turns, points_per_turn=60):
        """Generate Archimedean spiral points."""
        points = []
        total_points = int(turns * points_per_turn)
        for i in range(total_points):
            t = i / points_per_turn
            r = max_radius * t / turns
            angle = 2 * math.pi * t
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            points.append((x, y))
        return points


# ─────────────────────────────────────────────
# 2. GRAPH-BASED PATTERN REPRESENTATION
# ─────────────────────────────────────────────

class PatternGraph:
    """Represent Rangoli patterns as graphs for analysis and manipulation."""

    def __init__(self):
        self.graph = nx.Graph()

    def build_from_points(self, points, connection_radius=None):
        """Build graph from pattern points, connecting nearby nodes."""
        for i, pt in enumerate(points):
            self.graph.add_node(i, pos=pt)

        if connection_radius is None:
            if len(points) >= 4:
                tri = Delaunay(np.array(points))
                for simplex in tri.simplices:
                    for j in range(3):
                        n1, n2 = simplex[j], simplex[(j + 1) % 3]
                        dist = math.dist(points[n1], points[n2])
                        self.graph.add_edge(n1, n2, weight=dist)
            return

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = math.dist(points[i], points[j])
                if dist <= connection_radius:
                    self.graph.add_edge(i, j, weight=dist)

    def detect_symmetry_order(self):
        """Estimate rotational symmetry order from graph structure."""
        if len(self.graph.nodes) == 0:
            return 1
        degrees = [d for _, d in self.graph.degree()]
        if not degrees:
            return 1
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        most_common = max(degree_counts.values())
        for order in [8, 6, 4, 3, 2]:
            if most_common % order == 0:
                return order
        return 1

    def get_edges_as_lines(self):
        """Return edges as line segments for drawing."""
        positions = nx.get_node_attributes(self.graph, 'pos')
        lines = []
        for u, v in self.graph.edges():
            lines.append((positions[u], positions[v]))
        return lines

    def get_graph_stats(self):
        """Return graph statistics for analysis display."""
        if len(self.graph.nodes) == 0:
            return {"nodes": 0, "edges": 0, "density": 0, "avg_degree": 0}

        return {
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "density": round(nx.density(self.graph), 4),
            "avg_degree": round(sum(d for _, d in self.graph.degree()) / len(self.graph.nodes), 2),
            "connected_components": nx.number_connected_components(self.graph),
            "symmetry_order": self.detect_symmetry_order()
        }


# ─────────────────────────────────────────────
# 3. IMAGE ANALYSIS ENGINE
# ─────────────────────────────────────────────

class RangoliAnalyzer:
    """Analyze uploaded Rangoli images for symmetry, colors, and patterns."""

    @staticmethod
    def analyze_symmetry(image_array):
        """Detect rotational and reflective symmetry in an image."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape
        center = (w // 2, h // 2)

        symmetry_scores = {}
        for fold in [2, 3, 4, 5, 6, 8]:
            angle = 360.0 / fold
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(gray, M, (w, h))

            # Compute similarity between original and rotated
            diff = cv2.absdiff(gray, rotated)
            score = 1.0 - (np.mean(diff) / 255.0)
            symmetry_scores[f"{fold}-fold"] = round(score * 100, 1)

        # Reflective symmetry (horizontal & vertical)
        h_flip = cv2.flip(gray, 1)
        v_flip = cv2.flip(gray, 0)
        h_score = 1.0 - (np.mean(cv2.absdiff(gray, h_flip)) / 255.0)
        v_score = 1.0 - (np.mean(cv2.absdiff(gray, v_flip)) / 255.0)
        symmetry_scores["horizontal_reflection"] = round(h_score * 100, 1)
        symmetry_scores["vertical_reflection"] = round(v_score * 100, 1)

        best_fold = max(
            {k: v for k, v in symmetry_scores.items() if "fold" in k},
            key=lambda k: symmetry_scores[k]
        )
        symmetry_scores["dominant_symmetry"] = best_fold

        return symmetry_scores

    @staticmethod
    def extract_color_palette(image_array, n_colors=6):
        """Extract dominant color palette using K-Means clustering."""
        img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)

        # Sample pixels for speed
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)

        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(labels) * 100

        palette = []
        for color, pct in sorted(zip(colors, percentages), key=lambda x: -x[1]):
            r, g, b = color
            palette.append({
                "rgb": f"rgb({r},{g},{b})",
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "percentage": round(pct, 1)
            })
        return palette

    @staticmethod
    def detect_pattern_features(image_array):
        """Detect geometric features — lines, circles, contours."""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        num_lines = len(lines) if lines is not None else 0

        # Detect circles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=50, param2=30, minRadius=10, maxRadius=200)
        num_circles = len(circles[0]) if circles is not None else 0

        # Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Classify contour shapes
        shape_counts = {"triangles": 0, "quadrilaterals": 0, "pentagons": 0,
                        "hexagons": 0, "circles": 0, "complex": 0}
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            n = len(approx)
            if n == 3:
                shape_counts["triangles"] += 1
            elif n == 4:
                shape_counts["quadrilaterals"] += 1
            elif n == 5:
                shape_counts["pentagons"] += 1
            elif n == 6:
                shape_counts["hexagons"] += 1
            elif n > 6:
                circularity = 4 * math.pi * cv2.contourArea(cnt) / (peri * peri)
                if circularity > 0.7:
                    shape_counts["circles"] += 1
                else:
                    shape_counts["complex"] += 1

        # Compute complexity score
        total_features = num_lines + num_circles + len(contours)
        complexity = min(100, int(total_features / 5))

        return {
            "lines_detected": num_lines,
            "circles_detected": num_circles,
            "total_contours": len(contours),
            "shape_analysis": shape_counts,
            "edge_density": round(np.count_nonzero(edges) / edges.size * 100, 2),
            "complexity_score": complexity,
            "pattern_type": RangoliAnalyzer._classify_pattern(shape_counts, num_circles, num_lines)
        }

    @staticmethod
    def _classify_pattern(shapes, circles, lines):
        """Classify the Rangoli pattern type based on features."""
        if circles > 5:
            return "Mandala / Circular Kolam"
        elif shapes["hexagons"] > 3:
            return "Hexagonal Rangoli"
        elif lines > 20:
            return "Geometric Line Kolam"
        elif shapes["complex"] > 5:
            return "Freeform Floral Rangoli"
        else:
            return "Traditional Dot Rangoli"


# ─────────────────────────────────────────────
# 4. PATTERN GENERATION (Algorithmic AI)
# ─────────────────────────────────────────────

def generate_rangoli_image(params):
    """
    Generate a Rangoli pattern image based on parameters.
    Returns base64-encoded PNG image.
    """
    size = 800
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    engine = RangoliPatternEngine(size)

    style = params.get('style', 'mandala')
    symmetry = int(params.get('symmetry', 8))
    layers = int(params.get('layers', 5))
    color_scheme = params.get('colorScheme', 'traditional')
    complexity = int(params.get('complexity', 50))

    # Color palettes
    palettes = {
        'traditional': ['#FF6B35', '#F7C948', '#E63946', '#2D6A4F', '#FFFFFF',
                         '#D4A017', '#8B0000', '#FF4500'],
        'modern': ['#667EEA', '#764BA2', '#F093FB', '#4FACFE', '#00F2FE',
                    '#43E97B', '#FA709A', '#FEE140'],
        'earthy': ['#8D6E63', '#D4A76A', '#A1887F', '#BCAAA4', '#795548',
                    '#FFB74D', '#FF8A65', '#FFCC80'],
        'festive': ['#FF1744', '#FFD600', '#00E676', '#2979FF', '#FF9100',
                     '#D500F9', '#00B0FF', '#FFEA00'],
        'pastel': ['#FFB5E8', '#B5DEFF', '#E7FFAC', '#FFC9DE', '#C5A3FF',
                    '#AFF8DB', '#FFABAB', '#FFDAC1']
    }
    colors = palettes.get(color_scheme, palettes['traditional'])

    cx, cy = size // 2, size // 2
    max_radius = size // 2 - 40

    if style == 'mandala':
        _draw_mandala(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity)
    elif style == 'kolam':
        _draw_kolam(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity)
    elif style == 'floral':
        _draw_floral(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity)
    elif style == 'geometric':
        _draw_geometric(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity)
    elif style == 'peacock':
        _draw_peacock(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity)

    # Build graph representation
    pg = PatternGraph()
    radial_pts = engine.generate_radial_points(symmetry, layers, max_radius)
    pg.build_from_points(radial_pts)
    graph_stats = pg.get_graph_stats()

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return img_base64, graph_stats


def _draw_mandala(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity):
    """Draw a mandala-style Rangoli pattern."""
    # Outer decorative ring
    for i in range(3):
        r = max_radius - i * 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colors[0], width=2)

    # Concentric petal layers
    for layer in range(layers, 0, -1):
        radius = max_radius * layer / layers * 0.9
        num_petals = symmetry + (layers - layer) * 2
        color_idx = layer % len(colors)

        for p in range(num_petals):
            angle = 2 * math.pi * p / num_petals
            # Petal shape using ellipse
            petal_len = radius * 0.35
            petal_w = radius * 0.12

            # Petal tip
            tip_x = cx + radius * math.cos(angle)
            tip_y = cy + radius * math.sin(angle)
            # Petal base
            base_x = cx + (radius - petal_len) * math.cos(angle)
            base_y = cy + (radius - petal_len) * math.sin(angle)
            # Side points
            perp_angle = angle + math.pi / 2
            s1x = cx + (radius - petal_len * 0.5) * math.cos(angle) + petal_w * math.cos(perp_angle)
            s1y = cy + (radius - petal_len * 0.5) * math.sin(angle) + petal_w * math.sin(perp_angle)
            s2x = cx + (radius - petal_len * 0.5) * math.cos(angle) - petal_w * math.cos(perp_angle)
            s2y = cy + (radius - petal_len * 0.5) * math.sin(angle) - petal_w * math.sin(perp_angle)

            petal_points = [(base_x, base_y), (s1x, s1y), (tip_x, tip_y), (s2x, s2y)]
            draw.polygon(petal_points, fill=colors[color_idx], outline=colors[(color_idx + 1) % len(colors)])

        # Inner ring for this layer
        inner_r = radius * 0.65
        draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                     outline=colors[(color_idx + 2) % len(colors)], width=2)

        # Dots between petals
        if complexity > 30:
            dot_r = max(2, int(radius * 0.025))
            for p in range(num_petals):
                angle = 2 * math.pi * (p + 0.5) / num_petals
                dx = cx + (radius - petal_len * 0.5) * math.cos(angle)
                dy = cy + (radius - petal_len * 0.5) * math.sin(angle)
                draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r],
                             fill=colors[(color_idx + 3) % len(colors)])

    # Center motif
    cr = max_radius * 0.08
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=colors[0], outline=colors[1], width=2)
    cr2 = cr * 0.5
    draw.ellipse([cx - cr2, cy - cr2, cx + cr2, cy + cr2], fill=colors[1])


def _draw_kolam(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity):
    """Draw a Kolam-style pattern with interlocking curves."""
    # Dot grid
    grid_size = max(3, min(9, layers + 2))
    spacing = max_radius * 2 / (grid_size + 1)
    dots = engine.generate_dot_grid(grid_size, grid_size, spacing)

    # Draw connecting curves between dots
    for i, dot in enumerate(dots):
        for j, other in enumerate(dots):
            if i >= j:
                continue
            dist = math.dist(dot, other)
            if dist <= spacing * 1.5:
                # Draw curved line between dots
                mid_x = (dot[0] + other[0]) / 2
                mid_y = (dot[1] + other[1]) / 2
                # Add curve offset
                offset = spacing * 0.2 * (1 if (i + j) % 2 == 0 else -1)
                perp_angle = math.atan2(other[1] - dot[1], other[0] - dot[0]) + math.pi / 2
                ctrl_x = mid_x + offset * math.cos(perp_angle)
                ctrl_y = mid_y + offset * math.sin(perp_angle)

                color = colors[i % len(colors)]
                # Draw as polyline approximation of bezier
                pts = []
                for t in np.linspace(0, 1, 20):
                    bx = (1 - t) ** 2 * dot[0] + 2 * (1 - t) * t * ctrl_x + t ** 2 * other[0]
                    by = (1 - t) ** 2 * dot[1] + 2 * (1 - t) * t * ctrl_y + t ** 2 * other[1]
                    pts.append((bx, by))
                if len(pts) >= 2:
                    draw.line(pts, fill=color, width=3)

    # Radial decorations
    for fold in range(symmetry):
        angle = 2 * math.pi * fold / symmetry
        for layer in range(1, layers + 1):
            r = max_radius * layer / layers * 0.85
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            loop_r = spacing * 0.3
            color = colors[(fold + layer) % len(colors)]
            draw.ellipse([x - loop_r, y - loop_r, x + loop_r, y + loop_r],
                         outline=color, width=2)

    # Draw dots on top
    dot_r = max(3, int(spacing * 0.06))
    for dot in dots:
        draw.ellipse([dot[0] - dot_r, dot[1] - dot_r, dot[0] + dot_r, dot[1] + dot_r],
                     fill='white', outline=colors[0], width=1)

    # Center ornament
    cr = max_radius * 0.06
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=colors[0])


def _draw_floral(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity):
    """Draw a floral Rangoli with lotus and petal motifs."""
    # Background circles
    for layer in range(layers, 0, -1):
        r = max_radius * layer / layers
        color = colors[layer % len(colors)]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=1)

    # Multi-layer flower petals
    for layer in range(layers, 0, -1):
        radius = max_radius * layer / layers * 0.85
        num_petals = symmetry
        rotation = 360 / (symmetry * 2) * (layer % 2)  # Alternate petal positions

        for p in range(num_petals):
            angle = math.radians(360 * p / num_petals + rotation)
            color = colors[(p + layer) % len(colors)]

            # Large teardrop petal
            petal_len = radius * 0.5
            petal_w = radius * 0.18

            tip_x = cx + radius * math.cos(angle)
            tip_y = cy + radius * math.sin(angle)
            base_x = cx + (radius - petal_len) * math.cos(angle)
            base_y = cy + (radius - petal_len) * math.sin(angle)

            perp = angle + math.pi / 2
            s1 = (base_x + petal_w * 0.8 * math.cos(perp), base_y + petal_w * 0.8 * math.sin(perp))
            s2 = (base_x - petal_w * 0.8 * math.cos(perp), base_y - petal_w * 0.8 * math.sin(perp))
            m1_x = cx + (radius - petal_len * 0.3) * math.cos(angle) + petal_w * math.cos(perp)
            m1_y = cy + (radius - petal_len * 0.3) * math.sin(angle) + petal_w * math.sin(perp)
            m2_x = cx + (radius - petal_len * 0.3) * math.cos(angle) - petal_w * math.cos(perp)
            m2_y = cy + (radius - petal_len * 0.3) * math.sin(angle) - petal_w * math.sin(perp)

            petal = [s1, (m1_x, m1_y), (tip_x, tip_y), (m2_x, m2_y), s2]
            draw.polygon(petal, fill=color, outline='white')

            # Inner petal line
            draw.line([(base_x, base_y), (tip_x, tip_y)], fill='white', width=1)

        # Decorative dots ring
        if complexity > 25:
            dot_ring_r = radius * 0.7
            for d in range(num_petals * 2):
                da = 2 * math.pi * d / (num_petals * 2)
                dx = cx + dot_ring_r * math.cos(da)
                dy = cy + dot_ring_r * math.sin(da)
                dr = max(2, int(radius * 0.02))
                draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill='white')

    # Center lotus
    cr = max_radius * 0.12
    for i in range(3):
        r = cr - i * (cr / 4)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=colors[i % len(colors)], outline='white', width=2)


def _draw_geometric(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity):
    """Draw a geometric pattern Rangoli with polygons and stars."""
    # Star patterns
    for layer in range(layers, 0, -1):
        radius = max_radius * layer / layers * 0.9
        inner_radius = radius * 0.5
        num_points = symmetry
        color = colors[layer % len(colors)]

        # Star polygon
        star_pts = []
        for i in range(num_points * 2):
            angle = math.pi * i / num_points - math.pi / 2
            r = radius if i % 2 == 0 else inner_radius
            star_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(star_pts, outline=color, width=2)
        if layer % 2 == 0:
            draw.polygon(star_pts, fill=color + '40')

        # Connecting lines
        for i in range(0, num_points * 2, 2):
            opp = (i + num_points) % (num_points * 2)
            draw.line([star_pts[i], star_pts[opp]], fill=color, width=1)

        # Regular polygon inscribed
        poly_pts = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points - math.pi / 2
            r = radius * 0.7
            poly_pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(poly_pts, outline=colors[(layer + 1) % len(colors)], width=2)

    # Corner decorations
    if complexity > 40:
        for i in range(symmetry):
            angle = 2 * math.pi * i / symmetry
            ex = cx + max_radius * 0.95 * math.cos(angle)
            ey = cy + max_radius * 0.95 * math.sin(angle)
            dr = max_radius * 0.04
            draw.ellipse([ex - dr, ey - dr, ex + dr, ey + dr],
                         fill=colors[i % len(colors)], outline='white')

    # Center
    cr = max_radius * 0.08
    draw.regular_polygon((cx, cy, cr), symmetry, fill=colors[0], outline='white')


def _draw_peacock(draw, engine, cx, cy, max_radius, symmetry, layers, colors, complexity):
    """Draw a peacock-feather inspired Rangoli."""
    # Peacock feather eyes around the pattern
    for layer in range(layers, 0, -1):
        radius = max_radius * layer / layers * 0.85
        for i in range(symmetry):
            angle = 2 * math.pi * i / symmetry + (math.pi / symmetry) * (layer % 2)

            # Feather eye position
            fx = cx + radius * math.cos(angle)
            fy = cy + radius * math.sin(angle)
            eye_r = max_radius * 0.06 * (1 + layer / layers)

            # Outer feather shape (elongated towards center)
            feather_len = eye_r * 3
            tip_angle = angle + math.pi  # Points towards center
            tip_x = fx + feather_len * 0.5 * math.cos(tip_angle)
            tip_y = fy + feather_len * 0.5 * math.sin(tip_angle)
            tail_x = fx - feather_len * 0.5 * math.cos(tip_angle)
            tail_y = fy - feather_len * 0.5 * math.sin(tip_angle)

            perp = angle + math.pi / 2
            s1 = (fx + eye_r * math.cos(perp), fy + eye_r * math.sin(perp))
            s2 = (fx - eye_r * math.cos(perp), fy - eye_r * math.sin(perp))

            feather = [(tail_x, tail_y), s1, (tip_x, tip_y), s2]
            draw.polygon(feather, fill=colors[(i + layer) % len(colors)],
                         outline=colors[(i + layer + 1) % len(colors)])

            # Eye circles
            for j, c in enumerate([colors[2 % len(colors)], colors[0], 'white']):
                r = eye_r * (1 - j * 0.3)
                draw.ellipse([fx - r, fy - r, fx + r, fy + r], fill=c)

    # Spiral decorations
    if complexity > 30:
        for i in range(symmetry):
            angle = 2 * math.pi * i / symmetry
            spiral_pts = engine.generate_spiral(cx, cy, max_radius * 0.3, 1.5, 30)
            # Rotate spiral
            rotated = []
            for sx, sy in spiral_pts:
                dx, dy = sx - cx, sy - cy
                rx = dx * math.cos(angle) - dy * math.sin(angle) + cx
                ry = dx * math.sin(angle) + dy * math.cos(angle) + cy
                rotated.append((rx, ry))
            if len(rotated) >= 2:
                draw.line(rotated, fill=colors[i % len(colors)], width=2)

    # Center ornament
    cr = max_radius * 0.1
    for i in range(4):
        r = cr - i * (cr / 5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=colors[i % len(colors)], outline='white', width=1)


# ─────────────────────────────────────────────
# 5. FLASK ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """Health-check endpoint for load balancers and uptime monitors."""
    return jsonify({"status": "ok", "service": "rangoli-pattern-ai"})


@app.route('/api/generate', methods=['POST'])
def generate_pattern():
    """Generate a Rangoli pattern based on parameters."""
    params = request.json
    try:
        img_base64, graph_stats = generate_rangoli_image(params)
        return jsonify({
            "success": True,
            "image": img_base64,
            "graph_analysis": graph_stats,
            "parameters": params
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """Analyze an uploaded Rangoli image."""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    try:
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Could not read image"}), 400

        # Resize for consistent analysis
        target_size = 500
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Run analyses
        symmetry = RangoliAnalyzer.analyze_symmetry(img)
        palette = RangoliAnalyzer.extract_color_palette(img)
        features = RangoliAnalyzer.detect_pattern_features(img)

        # Generate edge visualization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        _, edge_buf = cv2.imencode('.png', edges)
        edge_b64 = base64.b64encode(edge_buf).decode('utf-8')

        return jsonify({
            "success": True,
            "symmetry_analysis": symmetry,
            "color_palette": palette,
            "pattern_features": features,
            "edge_map": edge_b64
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/styles', methods=['GET'])
def get_styles():
    """Return available pattern styles and parameters."""
    return jsonify({
        "styles": [
            {"id": "mandala", "name": "Mandala", "description": "Circular symmetric pattern with concentric layers"},
            {"id": "kolam", "name": "Kolam", "description": "Traditional South Indian dot-grid pattern with curves"},
            {"id": "floral", "name": "Floral", "description": "Lotus and petal-based decorative pattern"},
            {"id": "geometric", "name": "Geometric", "description": "Star and polygon-based mathematical pattern"},
            {"id": "peacock", "name": "Peacock", "description": "Peacock feather-inspired ornamental pattern"}
        ],
        "color_schemes": [
            {"id": "traditional", "name": "Traditional", "colors": ["#FF6B35", "#F7C948", "#E63946", "#2D6A4F"]},
            {"id": "modern", "name": "Modern", "colors": ["#667EEA", "#764BA2", "#F093FB", "#4FACFE"]},
            {"id": "earthy", "name": "Earthy", "colors": ["#8D6E63", "#D4A76A", "#A1887F", "#795548"]},
            {"id": "festive", "name": "Festive", "colors": ["#FF1744", "#FFD600", "#00E676", "#2979FF"]},
            {"id": "pastel", "name": "Pastel", "colors": ["#FFB5E8", "#B5DEFF", "#E7FFAC", "#C5A3FF"]}
        ],
        "symmetry_options": [3, 4, 5, 6, 8, 10, 12],
        "max_layers": 8
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum upload size is 16 MB."}), 413


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
