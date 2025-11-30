import mcdc
import numpy as np


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
cyl_z = mcdc.Surface.CylinderZ(center=np.array([0.0, 0.0]), radius=1.0)
# Surfaces for the assembly
cyl_x = mcdc.Surface.CylinderX(center=np.array([0.0, 0.0]), radius=1.0)
top_x = mcdc.Surface.PlaneX(x=2.5)
bot_x = mcdc.Surface.PlaneX(x=-2.5)
top_z = mcdc.Surface.PlaneZ(z=2.5)
bot_z = mcdc.Surface.PlaneZ(z=-2.5)
sphere = mcdc.Surface.Sphere(center=np.array([0.0, 0.0, 0.0]), radius=3.0)
cover_cell = mcdc.Cell(region=-sphere & ((+cyl_z | -bot_z | +top_z) & (+cyl_x | -bot_x | +top_x)), fill=cover)
# Cells for the assembly
fuel_cell = mcdc.Cell(region=((-cyl_z & +bot_z & -top_z) | (-cyl_x & +bot_x & -top_x)), fill=fuel)
water_cell = mcdc.Cell(region=+sphere, fill=water)
# Universe containing the fuel, cover, and water cells

assembly = mcdc.Universe(cells=[cover_cell, fuel_cell, water_cell])

vac_x_neg = mcdc.Surface.PlaneX(x=-10.0, boundary_condition='vacuum')

vac_x_pos = mcdc.Surface.PlaneX(x=10.0, boundary_condition='vacuum')

vac_y_neg = mcdc.Surface.PlaneY(y=-5.0, boundary_condition='vacuum')

vac_y_pos = mcdc.Surface.PlaneY(y=5.0, boundary_condition='vacuum')

vac_z_neg = mcdc.Surface.PlaneZ(z=-5.0, boundary_condition='vacuum')

vac_z_pos = mcdc.Surface.PlaneZ(z=5.0, boundary_condition='vacuum')

plane_x_zero = mcdc.Surface.PlaneX(x=0.0)

left_half_cell = mcdc.Cell(region=+vac_x_neg & -plane_x_zero & +vac_y_neg & -vac_y_pos & +vac_z_neg & -vac_z_pos, fill=assembly, translation=np.array([-5.0, 0.0, 0.0]))

right_half_cell = mcdc.Cell(region=+plane_x_zero & -vac_x_pos & +vac_y_neg & -vac_y_pos & +vac_z_neg & -vac_z_pos, fill=assembly, translation=np.array([5.0, 0.0, 0.0]), rotation=np.array([0.0, 10.0, 0.0]))

# Universe containing the left and right half cells as the root

mcdc.simulation.set_root_universe(cells=[left_half_cell, right_half_cell])

mcdc.Source(isotropic=True, energy_group=0, x=np.array([-0.1, 0.1]))

# Structured mesh for fission tally

fission_mesh = mcdc.MeshStructured(x=np.linspace(-10, 10, 201), y=np.array([0.0]), z=np.linspace(-5, 5, 101))

# Fission mesh tally

mcdc.TallyMesh(scores=['fission'], mesh=fission_mesh)

mcdc.settings.N_particle = 100

mcdc.settings.N_batch = 2

mcdc.settings.active_bank_buff = 1000

# === RUN ===

mcdc.run()