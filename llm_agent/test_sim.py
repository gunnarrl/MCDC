import mcdc
import numpy as np

fuel = mcdc.Material(
                    nuclide_composition={
                        'U235': 7.1122505549385897e-04,
        'U238': 2.2705807074495477e-02,
        'O16': 4.6720257483826924e-02,
                    }
                )
                # NOTE: CE mode requires MCDC_XSLIB environment variable
                # export MCDC_XSLIB="/path/to/your/nuclear/data"
                
moderator = mcdc.Material(
                    nuclide_composition={
                        'H1': 6.6845181611401544e-02,
        'O16': 3.3346567304434908e-02,
                    }
                )
                # NOTE: CE mode requires MCDC_XSLIB environment variable
                # export MCDC_XSLIB="/path/to/your/nuclear/data"
                
sphere_surface = mcdc.Surface.Sphere(center=[0.0, 0.0, 0.0], radius=10.0)
water_cell = mcdc.Cell(region=-sphere_surface, fill=moderator)
mcdc.Source(x=0.0, y=0.0, z=0.0, energy=1e6, isotropic=True)
mesh = mcdc.MeshUniform(x=(-10.0, 10.0, 20), y=(-10.0, 10.0, 20), z=(-10.0, 10.0, 20))
mcdc.TallyMesh(mesh=mesh, scores=['flux'])
mcdc.settings.N_particle = 10000
mcdc.settings.N_batch = 10
mcdc.run()
