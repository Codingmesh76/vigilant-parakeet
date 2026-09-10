# Compatibility note:
# This is a "legacy" (bl_info-based) single-file add-on. It relies only on
# long-stable, low-level Blender APIs (bmesh.ops, bpy.types.Operator/Panel/
# PropertyGroup/UIList, annotation-style bpy.props, VIEW3D_MT_mesh_add), none
# of which were touched by the 4.2-5.2 API changes, so it runs unmodified on
# Blender 5.2.1 LTS. Blender 4.2+ replaced single-.py drag-and-drop with the
# Extensions Platform for NEW add-ons, but legacy bl_info add-ons like this
# one are still fully supported - just install them via:
#   Edit > Preferences > Get Extensions > "..." menu (top right) >
#   "Install Legacy Add-on" > select this .py file > enable it.
# (Plain drag-and-drop into the 5.x window expects a manifest-based
# Extension .zip and will not work for a loose legacy .py file.)

bl_info = {
    "name": "Parametric History",
    "author": "Claude-Mahesh",
    "version": (1, 5, 1),
    "blender": (3, 0, 0),
    "location": "View3D > N-Panel > Param History, Add > Mesh",
    "description": (
        "Create primitive mesh objects whose creation parameters "
        "(radius, height, cap type, segments, etc.) stay editable and "
        "revertible forever, not just right after creation, with a live "
        "viewport preview while you edit and full support for keyframing "
        "those parameters like any standard Blender property (Graph Editor, "
        "Dope Sheet, drivers). "
        "Tested compatible through Blender 5.2 LTS."
    ),
    "category": "Add Mesh",
}

import bpy
import bmesh
import json
import time
from math import cos, sin, pi, radians
from mathutils import Matrix
from bpy.props import (
    StringProperty,
    FloatProperty,
    IntProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
    CollectionProperty,
)
from bpy.types import Operator, Panel, PropertyGroup, UIList
from bpy.app.handlers import persistent


# ---------------------------------------------------------------------------
# Primitive type registry
# ---------------------------------------------------------------------------

PRIM_TYPES = [
    ("CUBE", "Cube", "Add a cube"),
    ("PLANE", "Plane", "Add a plane"),
    ("GRID", "Grid", "Add a grid"),
    ("UV_SPHERE", "UV Sphere", "Add a UV sphere"),
    ("ICO_SPHERE", "Ico Sphere", "Add an ico sphere"),
    ("CYLINDER", "Cylinder", "Add a cylinder"),
    ("CONE", "Cone", "Add a cone"),
    ("CIRCLE", "Circle", "Add a circle"),
    ("TORUS", "Torus", "Add a torus"),
    ("MONKEY", "Monkey", "Add Suzanne"),
]

ICON_MAP = {
    "CUBE": "MESH_CUBE",
    "PLANE": "MESH_PLANE",
    "GRID": "MESH_GRID",
    "UV_SPHERE": "MESH_UVSPHERE",
    "ICO_SPHERE": "MESH_ICOSPHERE",
    "CYLINDER": "MESH_CYLINDER",
    "CONE": "MESH_CONE",
    "CIRCLE": "MESH_CIRCLE",
    "TORUS": "MESH_TORUS",
    "MONKEY": "MESH_MONKEY",
}

DEFAULTS = {
    "CUBE": {
        "box_width": 2.0, "box_length": 2.0, "box_height": 2.0,
        "box_width_segments": 1, "box_length_segments": 1, "box_height_segments": 1,
    },
    "PLANE": {"size": 2.0},
    "GRID": {"size": 2.0, "x_subdivisions": 10, "y_subdivisions": 10},
    "UV_SPHERE": {"radius": 1.0, "vertices": 32, "ring_count": 16},
    "ICO_SPHERE": {"radius": 1.0, "subdivisions": 2},
    "CYLINDER": {
        "radius": 1.0, "depth": 2.0, "vertices": 32, "cap_fill_type": "NGON",
        "depth_segments": 1, "cap_segments": 1,
    },
    "CONE": {
        "radius": 1.0, "radius2": 0.0, "depth": 2.0, "vertices": 32, "cap_fill_type": "NGON",
        "depth_segments": 1, "cap_segments": 1,
    },
    "CIRCLE": {"radius": 1.0, "vertices": 32, "cap_fill_type": "NGON", "side_segments": 1},
    "TORUS": {"radius": 1.0, "radius2": 0.25, "vertices": 48, "ring_count": 12},
    "MONKEY": {},
}

FIELD_NAMES = [
    "size", "radius", "radius2", "depth", "vertices", "ring_count",
    "subdivisions", "x_subdivisions", "y_subdivisions", "cap_fill_type",
    "depth_segments", "cap_segments", "side_segments",
    "box_width", "box_length", "box_height",
    "box_width_segments", "box_length_segments", "box_height_segments",
]

# Set to True while the add-on itself is bulk-assigning param fields (Add,
# Apply, Discard, Restore) so the live-preview update callback below doesn't
# fire redundantly for every single field during those internal writes.
_suppress_live_update = False


# ---------------------------------------------------------------------------
# Geometry building (bmesh, applied to the object's existing mesh datablock
# so the object itself, its modifiers, materials, and transform are kept)
# ---------------------------------------------------------------------------

def build_torus(bm, p):
    # Explicit ring-of-rings construction (major ring in the XY plane, tube
    # cross-section swept around it) - matches Blender's own torus topology
    # and avoids seam/winding issues that bmesh.ops.spin can produce.
    major_seg = max(3, int(p["vertices"]))
    minor_seg = max(3, int(p["ring_count"]))
    major_rad = p["radius"]
    minor_rad = p["radius2"]

    ring_verts = [[None] * minor_seg for _ in range(major_seg)]
    for i in range(major_seg):
        theta = 2.0 * pi * i / major_seg
        cos_t, sin_t = cos(theta), sin(theta)
        for j in range(minor_seg):
            phi = 2.0 * pi * j / minor_seg
            r = major_rad + minor_rad * cos(phi)
            x = r * cos_t
            y = r * sin_t
            z = minor_rad * sin(phi)
            ring_verts[i][j] = bm.verts.new((x, y, z))

    bm.verts.ensure_lookup_table()
    for i in range(major_seg):
        i2 = (i + 1) % major_seg
        for j in range(minor_seg):
            j2 = (j + 1) % minor_seg
            v00 = ring_verts[i][j]
            v10 = ring_verts[i2][j]
            v11 = ring_verts[i2][j2]
            v01 = ring_verts[i][j2]
            bm.faces.new((v00, v10, v11, v01))


def build_box(bm, p):
    # Six independently subdivided grids stitched together, rather than
    # bmesh.ops.create_cube (which has no per-axis segment control).
    hx = p["box_width"] / 2.0
    hy = p["box_length"] / 2.0
    hz = p["box_height"] / 2.0
    nx = max(1, int(p["box_width_segments"]))
    ny = max(1, int(p["box_length_segments"]))
    nz = max(1, int(p["box_height_segments"]))

    def face(seg_u, seg_v, half_u, half_v, rot_matrix, offset):
        ret = bmesh.ops.create_grid(bm, x_segments=seg_u, y_segments=seg_v, size=1.0)
        verts = ret["verts"]
        bmesh.ops.scale(bm, vec=(half_u, half_v, 1.0), verts=verts)
        if rot_matrix is not None:
            bmesh.ops.rotate(bm, cent=(0.0, 0.0, 0.0), matrix=rot_matrix, verts=verts)
        bmesh.ops.translate(bm, vec=offset, verts=verts)

    rot_x90 = Matrix.Rotation(radians(90.0), 3, "X")
    # Left/right faces need local-X -> world-Y and local-Y -> world-Z, which
    # is not reachable with a single axis-aligned 90-degree rotation - it's
    # the composition of a 90-degree X rotation followed by a 90-degree Z
    # rotation, i.e. the permutation (x, y, z) -> (z, x, y).
    rot_side = Matrix.Rotation(radians(90.0), 3, "Z") @ rot_x90

    # Top / bottom: span width (X) and length (Y)
    face(nx, ny, hx, hy, None, (0.0, 0.0, hz))
    face(nx, ny, hx, hy, None, (0.0, 0.0, -hz))
    # Front / back: span width (X) and height (Z)
    face(nx, nz, hx, hz, rot_x90, (0.0, hy, 0.0))
    face(nx, nz, hx, hz, rot_x90, (0.0, -hy, 0.0))
    # Left / right: span length (Y) and height (Z)
    face(ny, nz, hy, hz, rot_side, (hx, 0.0, 0.0))
    face(ny, nz, hy, hz, rot_side, (-hx, 0.0, 0.0))

    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)


def _build_cap(bm, ring_verts, radius, z, cap_type, cap_seg, radial_seg):
    if cap_type == "NOTHING":
        return
    if cap_type == "NGON":
        try:
            bm.faces.new(ring_verts)
        except ValueError:
            pass
        return

    # TRIFAN, optionally subdivided into concentric rings via cap_seg
    cap_seg = max(1, int(cap_seg))
    if cap_seg <= 1:
        center = bm.verts.new((0.0, 0.0, z))
        for i in range(radial_seg):
            i2 = (i + 1) % radial_seg
            try:
                bm.faces.new((center, ring_verts[i2], ring_verts[i]))
            except ValueError:
                pass
        return

    prev_ring = ring_verts
    for k in range(1, cap_seg):
        r_k = radius * (1.0 - k / cap_seg)
        new_ring = []
        for i in range(radial_seg):
            theta = 2.0 * pi * i / radial_seg
            new_ring.append(bm.verts.new((r_k * cos(theta), r_k * sin(theta), z)))
        for i in range(radial_seg):
            i2 = (i + 1) % radial_seg
            v00, v10 = prev_ring[i], prev_ring[i2]
            v11, v01 = new_ring[i2], new_ring[i]
            try:
                bm.faces.new((v00, v10, v11, v01))
            except ValueError:
                pass
        prev_ring = new_ring

    center = bm.verts.new((0.0, 0.0, z))
    for i in range(radial_seg):
        i2 = (i + 1) % radial_seg
        try:
            bm.faces.new((center, prev_ring[i2], prev_ring[i]))
        except ValueError:
            pass


def build_cylinder_cone(bm, radius1, radius2, depth, radial_seg, height_seg, cap_type, cap_seg):
    radial_seg = max(3, int(radial_seg))
    height_seg = max(1, int(height_seg))
    half = depth / 2.0

    rings = []  # list of (verts, is_point_apex)
    for h in range(height_seg + 1):
        t = h / height_seg
        z = -half + depth * t
        r = radius1 + (radius2 - radius1) * t
        if r <= 1e-9:
            rings.append(([bm.verts.new((0.0, 0.0, z))], True))
        else:
            verts = []
            for i in range(radial_seg):
                theta = 2.0 * pi * i / radial_seg
                verts.append(bm.verts.new((r * cos(theta), r * sin(theta), z)))
            rings.append((verts, False))

    bm.verts.ensure_lookup_table()

    for h in range(height_seg):
        va_list, a_pt = rings[h]
        vb_list, b_pt = rings[h + 1]
        if a_pt and b_pt:
            continue
        for i in range(radial_seg):
            i2 = (i + 1) % radial_seg
            try:
                if a_pt:
                    bm.faces.new((va_list[0], vb_list[i2], vb_list[i]))
                elif b_pt:
                    bm.faces.new((va_list[i], va_list[i2], vb_list[0]))
                else:
                    bm.faces.new((va_list[i], va_list[i2], vb_list[i2], vb_list[i]))
            except ValueError:
                pass

    bottom_verts, bottom_is_pt = rings[0]
    top_verts, top_is_pt = rings[-1]
    if not bottom_is_pt:
        _build_cap(bm, bottom_verts, radius1, -half, cap_type, cap_seg, radial_seg)
    if not top_is_pt:
        _build_cap(bm, top_verts, radius2, half, cap_type, cap_seg, radial_seg)


def build_circle(bm, p):
    # Built by hand (rather than bmesh.ops.create_circle) so the Triangle
    # Fan fill can be subdivided into concentric rings via Side Segments,
    # reusing the same disk-cap logic as the cylinder/cone caps.
    radial_seg = max(3, int(p["vertices"]))
    radius = p["radius"]
    side_seg = max(1, int(p["side_segments"]))
    fill_type = p["cap_fill_type"]

    ring = []
    for i in range(radial_seg):
        theta = 2.0 * pi * i / radial_seg
        ring.append(bm.verts.new((radius * cos(theta), radius * sin(theta), 0.0)))
    for i in range(radial_seg):
        i2 = (i + 1) % radial_seg
        bm.edges.new((ring[i], ring[i2]))

    _build_cap(bm, ring, radius, 0.0, fill_type, side_seg, radial_seg)


def build_mesh(bm, ptype, p):
    if ptype == "CUBE":
        build_box(bm, p)
    elif ptype == "PLANE":
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=p["size"] / 2.0)
    elif ptype == "GRID":
        bmesh.ops.create_grid(
            bm, x_segments=max(1, p["x_subdivisions"]), y_segments=max(1, p["y_subdivisions"]),
            size=p["size"] / 2.0,
        )
    elif ptype == "UV_SPHERE":
        bmesh.ops.create_uvsphere(
            bm, u_segments=max(3, p["vertices"]), v_segments=max(3, p["ring_count"]),
            radius=p["radius"],
        )
    elif ptype == "ICO_SPHERE":
        bmesh.ops.create_icosphere(bm, subdivisions=max(0, p["subdivisions"]), radius=p["radius"])
    elif ptype == "CYLINDER":
        build_cylinder_cone(
            bm, radius1=p["radius"], radius2=p["radius"], depth=p["depth"],
            radial_seg=p["vertices"], height_seg=p["depth_segments"],
            cap_type=p["cap_fill_type"], cap_seg=p["cap_segments"],
        )
    elif ptype == "CONE":
        build_cylinder_cone(
            bm, radius1=p["radius"], radius2=p["radius2"], depth=p["depth"],
            radial_seg=p["vertices"], height_seg=p["depth_segments"],
            cap_type=p["cap_fill_type"], cap_seg=p["cap_segments"],
        )
    elif ptype == "CIRCLE":
        build_circle(bm, p)
    elif ptype == "TORUS":
        build_torus(bm, p)
    elif ptype == "MONKEY":
        bmesh.ops.create_monkey(bm)


def regenerate_mesh(obj, ptype, params):
    old_mesh = obj.data

    # Capture current shading (Shade Smooth vs Shade Flat) so it survives
    # the mesh datablock being rebuilt from scratch. Note: "Shade Auto
    # Smooth" itself is a modifier ("Smooth by Angle") that lives on the
    # object, not the mesh, so it already survives untouched; only the
    # underlying smooth/flat face flag needs to be carried over here.
    was_smooth = False
    if old_mesh is not None and len(old_mesh.polygons) > 0:
        try:
            flags = [False] * len(old_mesh.polygons)
            old_mesh.polygons.foreach_get("use_smooth", flags)
            was_smooth = any(flags)
        except Exception:
            was_smooth = False

    bm = bmesh.new()
    build_mesh(bm, ptype, params)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    new_mesh = bpy.data.meshes.new(old_mesh.name if old_mesh else ptype.title())
    bm.to_mesh(new_mesh)
    bm.free()

    if was_smooth and len(new_mesh.polygons) > 0:
        try:
            new_mesh.polygons.foreach_set("use_smooth", [True] * len(new_mesh.polygons))
        except Exception:
            pass

    # Pre-4.1 Blender also stored auto-smooth settings directly on the mesh;
    # carry that over too for older files/compatibility.
    if old_mesh is not None and hasattr(old_mesh, "use_auto_smooth"):
        try:
            new_mesh.use_auto_smooth = old_mesh.use_auto_smooth
            new_mesh.auto_smooth_angle = old_mesh.auto_smooth_angle
        except Exception:
            pass

    obj.data = new_mesh
    if old_mesh is not None and old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)
    new_mesh.update()

    # Record what this rebuild reflects and force a redraw/re-evaluation.
    # Centralized here so every caller (manual edit, Apply/Cancel/Restore,
    # Add, and the frame-change handler) stays consistent automatically.
    try:
        obj["_ph_last_built"] = json.dumps(params, sort_keys=True)
    except Exception:
        pass
    try:
        obj.update_tag(refresh={"OBJECT", "DATA"})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Property groups
# ---------------------------------------------------------------------------

def on_draft_changed(self, context):
    """Live-preview update callback, fired whenever a draft field is edited
    directly (typing/dragging in the panel, or a driver/script setting it).
    Rebuilds the mesh immediately from the in-progress values without
    touching the applied (ph_current) state or history.

    Uses self.id_data (the Object that owns this property group) rather than
    context.object, since context.object is only correct when the edited
    object happens to be the active one - which is not guaranteed when a
    value changes via a driver or script on some other object."""
    if _suppress_live_update:
        return
    obj = self.id_data
    if not isinstance(obj, bpy.types.Object) or not obj.ph_type:
        return
    if not getattr(obj, "ph_live_preview", True):
        return
    try:
        regenerate_mesh(obj, obj.ph_type, params_to_dict(self))
    except Exception:
        pass


@persistent
def ph_frame_change_handler(scene, *_args):
    """Safety net for keyframe-driven animation: Blender's depsgraph does not
    always re-invoke a custom PropertyGroup field's update() callback when a
    value changes purely because of F-curve/keyframe evaluation (as opposed
    to a direct edit), so on every frame change we explicitly rebuild the
    mesh for any tracked object whose draft values may have been animated.

    Also force-tags changed objects and redraws open 3D viewports, since
    reassigning obj.data from inside this handler does not always propagate
    to the viewport on its own during playback."""
    changed_any = False
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not getattr(obj, "ph_type", "") or not getattr(obj, "ph_live_preview", True):
            continue
        try:
            current = json.dumps(params_to_dict(obj.ph_draft), sort_keys=True)
            if obj.get("_ph_last_built") == current:
                continue
            regenerate_mesh(obj, obj.ph_type, params_to_dict(obj.ph_draft))
            changed_any = True
        except Exception as e:
            print(f"[Parametric History] Frame-change rebuild failed for '{obj.name}': {e}")

    if changed_any:
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == "VIEW_3D":
                        area.tag_redraw()
        except Exception:
            pass


class PH_ParamsGroup(PropertyGroup):
    size: FloatProperty(name="Size", default=2.0, min=0.001, unit="LENGTH", update=on_draft_changed)
    radius: FloatProperty(name="Radius", default=1.0, min=0.0, unit="LENGTH", update=on_draft_changed)
    radius2: FloatProperty(name="Radius 2", default=0.0, min=0.0, unit="LENGTH", update=on_draft_changed)
    depth: FloatProperty(name="Depth", default=2.0, min=0.001, unit="LENGTH", update=on_draft_changed)
    vertices: IntProperty(name="Vertices", default=32, min=3, max=10000, update=on_draft_changed)
    ring_count: IntProperty(name="Rings", default=16, min=2, max=10000, update=on_draft_changed)
    subdivisions: IntProperty(name="Subdivisions", default=2, min=0, max=8, update=on_draft_changed)
    x_subdivisions: IntProperty(name="X Subdivisions", default=10, min=1, max=1000, update=on_draft_changed)
    y_subdivisions: IntProperty(name="Y Subdivisions", default=10, min=1, max=1000, update=on_draft_changed)
    depth_segments: IntProperty(name="Height Segments", default=1, min=1, max=1000, update=on_draft_changed)
    cap_segments: IntProperty(name="Cap Segments", default=1, min=1, max=1000, update=on_draft_changed)
    side_segments: IntProperty(name="Side Segments", default=1, min=1, max=1000, update=on_draft_changed)
    box_width: FloatProperty(name="Width", default=2.0, min=0.001, unit="LENGTH", update=on_draft_changed)
    box_length: FloatProperty(name="Length", default=2.0, min=0.001, unit="LENGTH", update=on_draft_changed)
    box_height: FloatProperty(name="Height", default=2.0, min=0.001, unit="LENGTH", update=on_draft_changed)
    box_width_segments: IntProperty(name="Width Segments", default=1, min=1, max=1000, update=on_draft_changed)
    box_length_segments: IntProperty(name="Length Segments", default=1, min=1, max=1000, update=on_draft_changed)
    box_height_segments: IntProperty(name="Height Segments", default=1, min=1, max=1000, update=on_draft_changed)
    cap_fill_type: EnumProperty(
        name="Cap/Fill Type",
        items=[("NGON", "Ngon", ""), ("TRIFAN", "Triangle Fan", ""), ("NOTHING", "Nothing", "")],
        default="NGON",
        update=on_draft_changed,
    )


class PH_HistoryItem(PropertyGroup):
    label: StringProperty(name="Label", default="")
    ptype: StringProperty(name="Type", default="")
    params_json: StringProperty(name="Params", default="{}")
    timestamp: StringProperty(name="Time", default="")


def params_to_dict(pg):
    return {f: getattr(pg, f) for f in FIELD_NAMES}


def apply_dict_to_params(pg, d):
    global _suppress_live_update
    prev = _suppress_live_update
    _suppress_live_update = True
    try:
        for f in FIELD_NAMES:
            if f in d:
                setattr(pg, f, d[f])
    finally:
        _suppress_live_update = prev


def copy_params(src_pg, dst_pg):
    apply_dict_to_params(dst_pg, params_to_dict(src_pg))


def push_history_entry(obj, label=""):
    item = obj.ph_history.add()
    item.ptype = obj.ph_type
    item.label = label or obj.ph_type.replace("_", " ").title()
    item.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    item.params_json = json.dumps(params_to_dict(obj.ph_current))
    obj.ph_history_index = len(obj.ph_history) - 1


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class OBJECT_OT_ph_add(Operator):
    bl_idname = "object.ph_add_primitive"
    bl_label = "Add Parametric Primitive"
    bl_options = {"REGISTER", "UNDO"}

    ptype: EnumProperty(items=[(t[0], t[1], t[2]) for t in PRIM_TYPES])

    def execute(self, context):
        ptype = self.ptype
        defaults = DEFAULTS.get(ptype, {})
        name = ptype.replace("_", " ").title()

        mesh = bpy.data.meshes.new(name=name)
        obj = bpy.data.objects.new(name=name, object_data=mesh)
        context.collection.objects.link(obj)
        obj.location = context.scene.cursor.location

        obj.ph_type = ptype
        obj.ph_live_preview = True
        apply_dict_to_params(obj.ph_current, defaults)
        apply_dict_to_params(obj.ph_draft, defaults)

        regenerate_mesh(obj, ptype, params_to_dict(obj.ph_current))

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        self.report({"INFO"}, f"Created parametric {name}")
        return {"FINISHED"}


class OBJECT_OT_ph_apply(Operator):
    bl_idname = "object.ph_apply"
    bl_label = "Apply"
    bl_description = "Keep the live-previewed shape: save the previous state to history and commit these values"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.ph_type)

    def execute(self, context):
        obj = context.object
        push_history_entry(obj, label=f"Edited {obj.ph_type.replace('_', ' ').title()}")
        copy_params(obj.ph_draft, obj.ph_current)
        regenerate_mesh(obj, obj.ph_type, params_to_dict(obj.ph_current))
        self.report({"INFO"}, "Shape updated")
        return {"FINISHED"}


class OBJECT_OT_ph_reset_draft(Operator):
    bl_idname = "object.ph_reset_draft"
    bl_label = "Cancel"
    bl_description = "Stop live-previewing: discard unapplied edits and rebuild the mesh from the last applied values"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.ph_type)

    def execute(self, context):
        obj = context.object
        copy_params(obj.ph_current, obj.ph_draft)
        regenerate_mesh(obj, obj.ph_type, params_to_dict(obj.ph_current))
        self.report({"INFO"}, "Reverted to last applied shape")
        return {"FINISHED"}


class OBJECT_OT_ph_convert_to_mesh(Operator):
    bl_idname = "object.ph_convert_to_mesh"
    bl_label = "Convert to Mesh"
    bl_description = (
        "Detach this object from Param History for good: it keeps its current "
        "geometry but becomes a standard mesh object, its parameter history is "
        "discarded, and it no longer appears in this panel"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.ph_type)

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        obj = context.object
        name = obj.ph_type.replace("_", " ").title()
        obj.ph_type = ""
        if "_ph_last_built" in obj:
            del obj["_ph_last_built"]
        obj.ph_history.clear()
        obj.ph_history_index = 0
        self.report({"INFO"}, f"{name} converted to a standard mesh object")
        return {"FINISHED"}


class OBJECT_OT_ph_restore_history(Operator):
    bl_idname = "object.ph_restore_history"
    bl_label = "Restore This Step"
    bl_description = "Revert the object to the parameters saved in the selected history entry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and bool(obj.ph_type) and len(obj.ph_history) > 0

    def execute(self, context):
        obj = context.object
        idx = obj.ph_history_index
        if idx < 0 or idx >= len(obj.ph_history):
            self.report({"WARNING"}, "No history entry selected")
            return {"CANCELLED"}

        entry = obj.ph_history[idx]
        push_history_entry(obj, label="Before revert")

        d = json.loads(entry.params_json)
        obj.ph_type = entry.ptype
        apply_dict_to_params(obj.ph_current, d)
        apply_dict_to_params(obj.ph_draft, d)
        regenerate_mesh(obj, obj.ph_type, d)
        self.report({"INFO"}, f"Restored: {entry.label}")
        return {"FINISHED"}


class OBJECT_OT_ph_history_remove(Operator):
    bl_idname = "object.ph_history_remove"
    bl_label = "Remove Entry"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and len(obj.ph_history) > 0

    def execute(self, context):
        obj = context.object
        idx = obj.ph_history_index
        if 0 <= idx < len(obj.ph_history):
            obj.ph_history.remove(idx)
            obj.ph_history_index = min(idx, len(obj.ph_history) - 1)
        return {"FINISHED"}


class OBJECT_OT_ph_history_move(Operator):
    bl_idname = "object.ph_history_move"
    bl_label = "Move Entry"
    bl_description = "Rearrange this history entry's position in the list"
    bl_options = {"REGISTER", "UNDO"}

    direction: EnumProperty(items=[("UP", "Up", ""), ("DOWN", "Down", "")])

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and len(obj.ph_history) > 1

    def execute(self, context):
        obj = context.object
        idx = obj.ph_history_index
        new_idx = idx - 1 if self.direction == "UP" else idx + 1
        if 0 <= new_idx < len(obj.ph_history):
            obj.ph_history.move(idx, new_idx)
            obj.ph_history_index = new_idx
        return {"FINISHED"}


class OBJECT_OT_ph_history_clear(Operator):
    bl_idname = "object.ph_history_clear"
    bl_label = "Clear History"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and len(obj.ph_history) > 0

    def execute(self, context):
        context.object.ph_history.clear()
        context.object.ph_history_index = 0
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class PH_UL_history(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{index + 1}. {item.label}", icon=ICON_MAP.get(item.ptype, "MESH_DATA"))
        row.label(text=item.timestamp)


def draw_type_fields(layout, pg, ptype):
    col = layout.column(align=True)
    if ptype == "CUBE":
        col.prop(pg, "box_width", text="Width")
        col.prop(pg, "box_length", text="Length")
        col.prop(pg, "box_height", text="Height")
        col.separator()
        col.prop(pg, "box_width_segments", text="Width Segments")
        col.prop(pg, "box_length_segments", text="Length Segments")
        col.prop(pg, "box_height_segments", text="Height Segments")
    elif ptype == "PLANE":
        col.prop(pg, "size")
    elif ptype == "GRID":
        col.prop(pg, "size")
        col.prop(pg, "x_subdivisions")
        col.prop(pg, "y_subdivisions")
    elif ptype == "UV_SPHERE":
        col.prop(pg, "radius")
        col.prop(pg, "vertices", text="Segments")
        col.prop(pg, "ring_count", text="Rings")
    elif ptype == "ICO_SPHERE":
        col.prop(pg, "radius")
        col.prop(pg, "subdivisions")
    elif ptype == "CYLINDER":
        col.prop(pg, "radius")
        col.prop(pg, "depth", text="Height")
        col.prop(pg, "vertices", text="Radius Segments")
        col.prop(pg, "depth_segments", text="Height Segments")
        col.prop(pg, "cap_fill_type", text="Cap Type")
        if pg.cap_fill_type == "TRIFAN":
            col.prop(pg, "cap_segments", text="Cap Segments")
    elif ptype == "CONE":
        col.prop(pg, "radius", text="Radius 1")
        col.prop(pg, "radius2", text="Radius 2")
        col.prop(pg, "depth", text="Height")
        col.prop(pg, "vertices", text="Radius Segments")
        col.prop(pg, "depth_segments", text="Height Segments")
        col.prop(pg, "cap_fill_type", text="Cap Type")
        if pg.cap_fill_type == "TRIFAN":
            col.prop(pg, "cap_segments", text="Cap Segments")
    elif ptype == "CIRCLE":
        col.prop(pg, "radius")
        col.prop(pg, "vertices", text="Radius Segments")
        col.prop(pg, "cap_fill_type", text="Fill Type")
        if pg.cap_fill_type == "TRIFAN":
            col.prop(pg, "side_segments", text="Side Segments")
    elif ptype == "TORUS":
        col.prop(pg, "radius", text="Major Radius")
        col.prop(pg, "radius2", text="Minor Radius")
        col.prop(pg, "vertices", text="Major Segments")
        col.prop(pg, "ring_count", text="Minor Segments")
    elif ptype == "MONKEY":
        col.label(text="No editable parameters", icon="INFO")


class VIEW3D_PT_ph_panel(Panel):
    bl_label = "Param History"
    bl_idname = "VIEW3D_PT_ph_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Param History"

    def draw(self, context):
        layout = self.layout
        obj = context.object

        box = layout.box()
        box.label(text="Create Primitive", icon="ADD")
        grid = box.grid_flow(row_major=True, columns=2, even_columns=True)
        for pid, pname, _ in PRIM_TYPES:
            op = grid.operator("object.ph_add_primitive", text=pname, icon=ICON_MAP[pid])
            op.ptype = pid

        if obj is None or obj.type != "MESH":
            layout.separator()
            layout.label(text="Select a mesh object", icon="INFO")
            return

        if not obj.ph_type:
            layout.separator()
            box = layout.box()
            box.label(text="No parametric history on this object.", icon="INFO")
            box.label(text="Create a primitive above to start")
            box.label(text="tracking its parameters.")
            return

        layout.separator()
        box = layout.box()
        header = box.row(align=True)
        header.label(text=f"Type: {obj.ph_type.replace('_', ' ').title()}", icon=ICON_MAP.get(obj.ph_type, "OBJECT_DATA"))
        header.prop(obj, "ph_live_preview", text="", icon="HIDE_OFF" if obj.ph_live_preview else "HIDE_ON")
        draw_type_fields(box, obj.ph_draft, obj.ph_type)
        hint = box.row()
        hint.label(text="Right-click any value above to Insert Keyframe", icon="KEYFRAME")

        if params_to_dict(obj.ph_draft) != params_to_dict(obj.ph_current):
            note = box.row()
            note.alert = True
            note.label(
                text="Unapplied live preview - Apply to keep or Cancel to revert",
                icon="ERROR",
            )

        row = box.row(align=True)
        row.operator("object.ph_apply", icon="CHECKMARK")
        row.operator("object.ph_reset_draft", icon="LOOP_BACK")
        box.operator("object.ph_convert_to_mesh", icon="MESH_DATA")

        layout.separator()
        box = layout.box()
        box.label(text=f"History ({len(obj.ph_history)})", icon="RECOVER_LAST")
        row = box.row()
        row.template_list("PH_UL_history", "", obj, "ph_history", obj, "ph_history_index", rows=4)
        col = row.column(align=True)
        col.operator("object.ph_history_move", text="", icon="TRIA_UP").direction = "UP"
        col.operator("object.ph_history_move", text="", icon="TRIA_DOWN").direction = "DOWN"
        col.separator()
        col.operator("object.ph_history_remove", text="", icon="X")

        row = box.row(align=True)
        row.operator("object.ph_restore_history", text="Restore Selected Step", icon="LOOP_BACK")
        box.operator("object.ph_history_clear", icon="TRASH")


def add_menu_func(self, context):
    layout = self.layout
    layout.separator()
    layout.label(text="Parametric History")
    for pid, pname, _ in PRIM_TYPES:
        op = layout.operator("object.ph_add_primitive", text=pname, icon=ICON_MAP[pid])
        op.ptype = pid


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    PH_ParamsGroup,
    PH_HistoryItem,
    PH_UL_history,
    OBJECT_OT_ph_add,
    OBJECT_OT_ph_apply,
    OBJECT_OT_ph_reset_draft,
    OBJECT_OT_ph_convert_to_mesh,
    OBJECT_OT_ph_restore_history,
    OBJECT_OT_ph_history_remove,
    OBJECT_OT_ph_history_move,
    OBJECT_OT_ph_history_clear,
    VIEW3D_PT_ph_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Object.ph_type = StringProperty(name="Parametric Type", default="")
    bpy.types.Object.ph_current = PointerProperty(type=PH_ParamsGroup)
    bpy.types.Object.ph_draft = PointerProperty(type=PH_ParamsGroup)
    bpy.types.Object.ph_history = CollectionProperty(type=PH_HistoryItem)
    bpy.types.Object.ph_history_index = IntProperty(default=0)
    bpy.types.Object.ph_live_preview = BoolProperty(
        name="Live Preview",
        description="Rebuild the mesh in the viewport immediately as you edit values below",
        default=True,
    )

    bpy.types.VIEW3D_MT_mesh_add.append(add_menu_func)

    if ph_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(ph_frame_change_handler)


def unregister():
    if ph_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(ph_frame_change_handler)

    bpy.types.VIEW3D_MT_mesh_add.remove(add_menu_func)

    del bpy.types.Object.ph_live_preview
    del bpy.types.Object.ph_history_index
    del bpy.types.Object.ph_history
    del bpy.types.Object.ph_draft
    del bpy.types.Object.ph_current
    del bpy.types.Object.ph_type

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
