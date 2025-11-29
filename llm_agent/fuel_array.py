import mcdc
import numpy as np


# === MATERIAL DEFINITIONS ===
cover = mcdc.MaterialMG(
    capture=np.array([0.05]),
    scatter=np.array([[0.95]])
)
fuel = mcdc.MaterialMG(
    capture=np.array([0.45]),
    fission=np.array([0.55]),
    nu_p=np.array([2.5])
)
water = mcdc.MaterialMG(
    capture=np.array([0.02]),
    scatter=np.array([[0.08]])
)

# === SURFACE DEFINITIONS ===
bot_x = mcdc.Surface.PlaneX(x=-2.5)
cyl_x = mcdc.Surface.CylinderX(center=np.array([0.0, 0.0]), radius=1.0)
top_z = mcdc.Surface.PlaneZ(z=2.5)
top_x = mcdc.Surface.PlaneX(x=2.5)
bot_z = mcdc.Surface.PlaneZ(z=-2.5)
cyl_z = mcdc.Surface.CylinderZ(center=np.array([0.0, 0.0]), radius=1.0)
sphere = mcdc.Surface.Sphere(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
# Three planes at -10, 0, 10
plane_neg10 = mcdc.Surface.PlaneX(x=-10.0, boundary_condition='vacuum')
plane_0 = mcdc.Surface.PlaneX(x=0.0)
plane_pos10 = mcdc.Surface.PlaneX(x=10.0, boundary_condition='vacuum')
plane_z_neg5 = mcdc.Surface.PlaneZ(z=-5.0, boundary_condition='vacuum')
plane_y_neg5 = mcdc.Surface.PlaneY(y=-5.0, boundary_condition='vacuum')
plane_z_pos5 = mcdc.Surface.PlaneZ(z=5.0, boundary_condition='vacuum')
plane_y_pos5 = mcdc.Surface.PlaneY(y=5.0, boundary_condition='vacuum')

# === CELL DEFINITIONS ===
# Cell inside sphere but outside shooting star
cover_cell = mcdc.Cell(region=-sphere & ((+cyl_z | -bot_z | +top_z) & (+cyl_x | -bot_x | +top_x)), fill=cover)
# Cell for the shooting star region
fuel_cell = mcdc.Cell(region=((-cyl_z & +bot_z & -top_z) | (-cyl_x & +bot_x & -top_x)), fill=fuel)
# Cell outside the sphere
water_cell = mcdc.Cell(region=+sphere, fill=water)

# === UNIVERSE DEFINITIONS ===
assembly = mcdc.Universe(cells=[cover_cell, fuel_cell, water_cell])

# Cells filling the left and right halves of a box
left_half_cell = mcdc.Cell(region=+plane_neg10 & -plane_0 & +plane_y_neg5 & -plane_y_pos5 & +plane_z_neg5 & -plane_z_pos5, fill=assembly, translation=np.array([-5.0, 0.0, 0.0]))
right_half_cell = mcdc.Cell(region=+plane_0 & -plane_pos10 & +plane_y_neg5 & -plane_y_pos5 & +plane_z_neg5 & -plane_z_pos5, fill=assembly, translation=np.array([5.0, 0.0, 0.0]), rotation=np.array([0.0, 10.0, 0.0]))

mcdc.simulation.set_root_universe(cells=[left_half_cell, right_half_cell])


# === SOURCE DEFINITIONS ===
mcdc.Source(x=np.array([-0.1, 0.1]), energy_group=0)

# === TALLY DEFINITIONS ===
# Tally mesh for fission
fission_mesh = mcdc.MeshStructured(x=np.linspace(-10, 10, 201), y=np.array([0.0]), z=np.linspace(-5, 5, 101))
mcdc.TallyMesh(scores=['fission'], mesh=fission_mesh)

# === SETTINGS DEFINITIONS ===
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000
# === RUN ===
mcdc.run()