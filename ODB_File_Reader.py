from __future__ import print_function
from odbAccess import *
from abaqusConstants import *
import csv, math

# -------------------------------
# User settings
# -------------------------------
ODB_PATH = 'D6WA65T20.odb'
STEP_NAME = 'ISF'
OUT_CSV  = 'D6WA65T20.csv'   # Python 2: will open in binary mode

# -------------------------------
# Helpers
# -------------------------------
def get_value_safe(v):
    """Return field value as a list (handles dataDouble/data and scalars)."""
    if hasattr(v, 'dataDouble'):
        data = v.dataDouble
    elif hasattr(v, 'data'):
        data = v.data
    else:
        return []
    try:
        return list(data)
    except TypeError:
        return [data]

def avg_vectors(a, b):
    """Element-wise add vector b into a (growing list), return new list."""
    if not a:
        return b[:]
    n = max(len(a), len(b))
    A = a[:] + [0.0]*(n - len(a))
    B = b[:] + [0.0]*(n - len(b))
    return [A[i] + B[i] for i in range(n)]

def pad_to(vec, n, pad=None):
    if vec is None:
        return [pad]*n
    vec = list(vec)
    if len(vec) < n:
        vec += [pad]*(n - len(vec))
    else:
        vec = vec[:n]
    return vec

def compute_von_mises_from_vec(svec):
    """Compute 3D von Mises from [S11,S22,S33,S12,S13,S23].
       Return None if insufficient/None entries."""
    if not svec or len(svec) < 6:
        return None
    try:
        s11, s22, s33, s12, s13, s23 = svec[0], svec[1], svec[2], svec[3], svec[4], svec[5]
        if any(v is None for v in (s11, s22, s33, s12, s13, s23)):
            return None
        s11, s22, s33, s12, s13, s23 = map(float, (s11, s22, s33, s12, s13, s23))
        vm = math.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2)
                       + 3.0 * (s12**2 + s23**2 + s13**2))
        return vm
    except Exception:
        return None

# -------------------------------
# Open ODB and pick frame
# -------------------------------
odb = openOdb(path=ODB_PATH)
step = odb.steps[STEP_NAME]
frame = step.frames[-1]   # last frame

assembly = odb.rootAssembly
instances = assembly.instances   # dict: name -> OdbInstance

# -------------------------------
# Collect nodal coordinates (all instances)
# -------------------------------
node_coords = {}  # key: (instName, nodeLabel) -> [X,Y,Z]
for iname, inst in instances.items():
    for node in inst.nodes:
        node_coords[(iname, node.label)] = [node.coordinates[0],
                                           node.coordinates[1],
                                           node.coordinates[2]]

# -------------------------------
# Nodal displacements (U)
# -------------------------------
disp_map = {}  # key: (instName, nodeLabel) -> [U1,U2,U3]
if 'U' in frame.fieldOutputs.keys():
    U = frame.fieldOutputs['U']
    for v in U.values:
        key = (v.instance.name, v.nodeLabel)
        vals = get_value_safe(v)
        # pad to 3
        while len(vals) < 3:
            vals.append(None)
        disp_map[key] = vals[:3]

# -------------------------------
# Element fields: accumulate sums and counts per element
# -------------------------------
elem_S_sum, elem_S_cnt = {}, {}
elem_PE_sum, elem_PE_cnt = {}, {}
elem_PEEQ_sum, elem_PEEQ_cnt = {}, {}
elem_STH_sum, elem_STH_cnt = {}, {}

# Stress S
if 'S' in frame.fieldOutputs.keys():
    S = frame.fieldOutputs['S']
    for v in S.values:
        key = (v.instance.name, v.elementLabel)
        vec = get_value_safe(v)
        elem_S_sum[key] = avg_vectors(elem_S_sum.get(key, []), vec)
        elem_S_cnt[key] = elem_S_cnt.get(key, 0) + 1

# Strain PE
if 'PE' in frame.fieldOutputs.keys():
    PE = frame.fieldOutputs['PE']
    for v in PE.values:
        key = (v.instance.name, v.elementLabel)
        vec = get_value_safe(v)
        elem_PE_sum[key] = avg_vectors(elem_PE_sum.get(key, []), vec)
        elem_PE_cnt[key] = elem_PE_cnt.get(key, 0) + 1

# Equivalent plastic strain PEEQ (scalar)
if 'PEEQ' in frame.fieldOutputs.keys():
    PEEQ = frame.fieldOutputs['PEEQ']
    for v in PEEQ.values:
        key = (v.instance.name, v.elementLabel)
        val = get_value_safe(v)
        if isinstance(val, (list, tuple)):
            val = val[0]
        try:
            elem_PEEQ_sum[key] = elem_PEEQ_sum.get(key, 0.0) + float(val)
            elem_PEEQ_cnt[key] = elem_PEEQ_cnt.get(key, 0) + 1
        except Exception:
            pass

# Shell thickness STH (scalar or per section point)
if 'STH' in frame.fieldOutputs.keys():
    STH = frame.fieldOutputs['STH']
    for v in STH.values:
        key = (v.instance.name, v.elementLabel)
        val = get_value_safe(v)
        if isinstance(val, (list, tuple)):
            val = val[0]
        try:
            elem_STH_sum[key] = elem_STH_sum.get(key, 0.0) + float(val)
            elem_STH_cnt[key] = elem_STH_cnt.get(key, 0) + 1
        except Exception:
            pass

# -------------------------------
# Average element fields
# -------------------------------
elem_S = {}
for k, vsum in elem_S_sum.items():
    n = float(elem_S_cnt.get(k, 1))
    elem_S[k] = [x / n for x in vsum]

elem_PE = {}
for k, vsum in elem_PE_sum.items():
    n = float(elem_PE_cnt.get(k, 1))
    elem_PE[k] = [x / n for x in vsum]

elem_PEEQ = {}
for k, s in elem_PEEQ_sum.items():
    n = float(elem_PEEQ_cnt.get(k, 1))
    elem_PEEQ[k] = s / n

elem_STH = {}
for k, s in elem_STH_sum.items():
    n = float(elem_STH_cnt.get(k, 1))
    elem_STH[k] = s / n

# -------------------------------
# Element centroids (all instances)
# -------------------------------
elem_centroid = {}  # key: (instName, elemLabel) -> [Xc,Yc,Zc]
for iname, inst in instances.items():
    for e in inst.elements:
        coords = []
        for nlabel in e.connectivity:
            node = inst.getNodeFromLabel(nlabel)
            coords.append(node.coordinates)
        if coords:
            sx = sum([c[0] for c in coords]) / float(len(coords))
            sy = sum([c[1] for c in coords]) / float(len(coords))
            sz = sum([c[2] for c in coords]) / float(len(coords))
        else:
            sx = sy = sz = 0.0
        elem_centroid[(iname, e.label)] = [sx, sy, sz]

# -------------------------------
# Write ONE combined CSV
# Note: Python 2.7 csv expects binary mode 'wb'; do NOT use newline=...
# -------------------------------
headers = ["Type", "Instance", "Label", "X", "Y", "Z",
           "U1", "U2", "U3",
           "S11", "S22", "S33", "S12", "S13", "S23",
           "S_Mises",
           "PE11", "PE22", "PE33", "PE12", "PE13", "PE23",
           "PEEQ",
           "Thickness"]

f = open(OUT_CSV, 'wb')   # Python2: binary mode for csv
writer = csv.writer(f)
writer.writerow(headers)

# Rows for NODES (node rows first)
total_cols = len(headers)
for (iname, nlabel), xyz in sorted(node_coords.items()):
    disp = disp_map.get((iname, nlabel), [None, None, None])
    # build full row: [Type, Instance, Label, X,Y,Z, U1,U2,U3] + padding None for remaining columns
    node_prefix = ["Node", iname, nlabel, xyz[0], xyz[1], xyz[2]]
    node_mid = [disp[0], disp[1], disp[2]]
    padding = [None] * (total_cols - (len(node_prefix) + len(node_mid)))
    row = node_prefix + node_mid + padding
    writer.writerow(row)

# Rows for ELEMENTS
for (iname, elabel), cxyz in sorted(elem_centroid.items()):
    Svec = pad_to(elem_S.get((iname, elabel)), 6, None)
    PEvec = pad_to(elem_PE.get((iname, elabel)), 6, None)
    peeq = elem_PEEQ.get((iname, elabel), None)
    sth  = elem_STH.get((iname, elabel), None)
    s_mises = compute_von_mises_from_vec(Svec)

    elem_prefix = ["Element", iname, elabel, cxyz[0], cxyz[1], cxyz[2]]
    # elements have no nodal U columns -> leave U slots empty
    elem_u_slots = [None, None, None]
    row_mid = Svec + [s_mises] + PEvec + [peeq, sth]
    row = elem_prefix + elem_u_slots + row_mid
    # ensure correct length
    if len(row) < total_cols:
        row += [None]*(total_cols - len(row))
    elif len(row) > total_cols:
        row = row[:total_cols]
    writer.writerow(row)

f.close()
print(" Combined CSV written to:", OUT_CSV)

odb.close()
