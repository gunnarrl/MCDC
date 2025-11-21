import mcdc
import numpy as np

# Pure fission material
fission_mat = mcdc.MaterialMG(
    fission=np.array([1.0]),
    nu_p=np.array([1.2])
)
# Pure scatter material
scatter_mat = mcdc.MaterialMG(
    scatter=np.array([[1.5]])
)
# Start of Cube
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition='vacuum')
s2 = mcdc.Surface.PlaneX(x=4.0, boundary_condition='vacuum')
s3 = mcdc.Surface.PlaneY(y=0.0, boundary_condition='vacuum')
s4 = mcdc.Surface.PlaneY(y=4.0, boundary_condition='vacuum')
s5 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition='vacuum')
s6 = mcdc.Surface.PlaneZ(z=4.0, boundary_condition='vacuum')
s_sphere = mcdc.Surface.Sphere(center=np.array([2, 2, 2]), radius=1.5)
# Cell: Fission material inside sphere
cell_sphere = mcdc.Cell(region=-s_sphere, fill=fission_mat)
# Cell: Scatter material inside cube, outside sphere
cell_cube_outer_sphere = mcdc.Cell(region=+s1 & -s2 & +s3 & -s4 & +s5 & -s6 & +s_sphere, fill=scatter_mat)
source = mcdc.Source(x=np.array([0, 4]), y=np.array([0, 4]), z=np.array([0, 4]), isotropic=True, time=np.array([0, 50]), energy_group=0)
# Uniform Mesh Tally
mesh = mcdc.MeshUniform(x=(0, 4, 1), y=(0, 4, 1), z=(0, 4, 1))
mesh_tally = mcdc.TallyMesh(scores=['fission'], mesh=mesh)
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2

mcdc.run()
