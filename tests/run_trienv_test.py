import numpy as np

from util import icosphere
from trimem.mc.trilmp import TriLmp, Beads
from trimesh import Trimesh

class SimulationManager():
    def __init__(self, sigma_membrane=1.0, resolution=2):

        self.sigma_membrane = sigma_membrane
        self.resolution = resolution

        vertices, faces = icosphere(resolution)

        self.mesh = Trimesh(vertices=vertices, faces=faces)
        # rescaling it so that we start from the right distances
        desired_average_distance = 2**(1.0/6.0) * sigma_membrane
        current_average_distance = np.mean(self.mesh.edges_unique_length)
        scaling = desired_average_distance/current_average_distance
        self.mesh.vertices *= scaling
        self.postequilibration_lammps_command = []

    @staticmethod
    def equilibriation_thermostat_command(initial_temperature, langevin_damp, langevin_seed: int, fix_gcmc: bool=True):
        commands = []

        if fix_gcmc:
            # necessary to include the nve because we have removed it
            commands.append(f'fix vertexnve vertices nve')
            commands.append(f'fix lvt vertices langevin {initial_temperature} {initial_temperature}  {langevin_damp} {langevin_seed} zero yes tally yes')
            return '\n'.join(commands)
            
        commands.append(f"fix lvt vertices langevin {initial_temperature} {initial_temperature}  {langevin_damp} {langevin_seed} zero yes")
        return '\n'.join(commands)
    
    @staticmethod
    def equilibriation_chemostat_command(n_region, N, X, typem, seedm, Tm, mu, maxp):
        f"fix mygcmc_{n_region} metabolites gcmc {N} {X} 0 {typem} {seedm} {Tm} {mu} 0 region gcmc_region_{n_region} max {maxp}"

    @staticmethod
    def langevin_commands(membrane_vertex_mass, initial_temperature: float, langevin_damp, langevin_seed: int, beads: Beads = None, length_scale: float = 1.):
        fix_gcmc = True

        commands = []
        sc0=membrane_vertex_mass/length_scale

        bds_info_lgv = ''

        if beads is not None:
            if beads.n_types>1:
                bds_info_lgv = ''
                for i in range(beads.n_types):
                    bds_info_lgv += f'scale {i + 2} {(beads.masses[i] / beads.bead_sizes[i])/sc0} '
            else:
                bds_info_lgv=f'scale 2 {(beads.masses / beads.bead_sizes)/sc0}'

            # FIX GCMC (scale the impact of the thermostat on the metabolites - beads masses is by default 1)
            if fix_gcmc:
                bds_info_lgv+= f'scale 2 {(beads.masses / beads.bead_sizes[0])/sc0}'
            

        commands.append("fix mynve all nve")
        commands.append(f"fix lvt all langevin {initial_temperature} {initial_temperature} {langevin_damp} {langevin_seed} zero yes {bds_info_lgv}")
        return '\n'.join(commands)

    def run(self):
        # membrane parameters
        membrane_vertex_mass = 1.0
        kappa_b=20.0
        kappa_a=2.5e5
        kappa_v=2.5e5
        kappa_c=0.0
        kappa_t=1.0e4
        kappa_r=1.0e4

        # trimem parameters
        traj_steps=50
        flip_ratio=0.1
        step_size=0.001
        total_sim_time=1000 # 00  # in time units
        discrete_snapshots=10   # in time units
        print_frequency = int(discrete_snapshots/(step_size*traj_steps))

        initial_temperature=1.0                    # MD PART SIMULATION: temperature of the system
        pure_MD=False,                             # MD PART SIMULATION: accept every MD trajectory?
        langevin_damp=1.0
        langevin_seed=123
        (xlo,xhi,ylo,yhi,zlo,zhi) = (-50, 50, -50, 50, -50, 50)
        switch_mode = 'random'

        # GCMC region
        variable_factor = 10
        N_gcmc_1=int(variable_factor*langevin_damp/step_size)
        X_gcmc_1=100
        seed_gcmc_1 = langevin_seed
        mu_gcmc_1=0
        vfrac = 0.01
        geometric_factor = 1.0
        sigma_metabolites = 1.0
        x_membrane_max = np.max(self.mesh.vertices[:, 0])
        r_mean = np.mean(
            np.linalg.norm(
                np.mean(self.mesh.vertices, axis=0)-self.mesh.vertices, axis=1
            )
        )
        height_width = r_mean*geometric_factor
        gcmc_xlo, gcmc_xhi = x_membrane_max, xhi, # xlo, xhi
        gcmc_ylo, gcmc_yhi = -height_width/2, height_width/2, # ylo, yhi
        gcmc_zlo, gcmc_zhi = -height_width/2, height_width/2, # zlo, zhi

        vtotal_region = (xhi-x_membrane_max)*(height_width)*(height_width)
        max_gcmc_1 = int((vfrac*vtotal_region*3)/(4*np.pi*(sigma_metabolites*0.5)**3))

        trilmp = TriLmp(
            initialize=True,                          # use mesh to initialize mesh reference
            mesh_points=self.mesh.vertices,           # input mesh vertices
            mesh_faces=self.mesh.faces,               # input of the mesh faces
            kappa_b=kappa_b,                          # MEMBRANE MECHANICS: bending modulus (kB T)
            kappa_a=kappa_a,                          # MEMBRANE MECHANICS: constraint on area change from target value (kB T)
            kappa_v=kappa_v,                          # MEMBRANE MECHANICS: constraint on volume change from target value (kB T)
            kappa_c=kappa_c,                          # MEMBRANE MECHANICS: constraint on area difference change (understand meaning) (kB T)
            kappa_t=kappa_t,                          # MEMBRANE MECHANICS: tethering potential to constrain edge length (kB T)
            kappa_r=kappa_r,                          # MEMBRANE MECHANICS: repulsive potential to prevent surface intersection (kB T)
            
            num_particle_types=2,                       # how many particle types will there be in the system
            mass_particle_type=[membrane_vertex_mass, 1],# the mass of the particle per type
            group_particle_type=['vertices', 'metabolites'],

            step_size=step_size,                      # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
            traj_steps=traj_steps,                    # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
            flip_ratio=flip_ratio,                    # MC PART SIMULATION: fraction of edges to flip?
            check_neigh_every=1,                      # NEIGHBOUR LISTS
            equilibration_rounds=1,                   # MEMBRANE EQUILIBRATION ROUNDS

            box=(xlo,xhi,ylo,yhi,zlo,zhi),

            output_prefix='data/data',                # OUTPUT: prefix for output filenames
            restart_prefix='data/data',               # OUTPUT: name for checkpoint files
            checkpoint_every=print_frequency,         # OUTPUT: interval of checkpoints (alternating pickles)
            output_format='lammps_txt',               # OUTPUT: choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
            output_counter=0,                         # OUTPUT: initialize trajectory number in writer class
            performance_increment=print_frequency,    # OUTPUT: output performace stats to prefix_performance.dat file
            energy_increment=print_frequency,         # OUTPUT: output energies to energies.dat file
            pure_MD=pure_MD,                          # MD PART SIMULATION: accept every MD trajectory?
        )

        # add a gcmc region
        trilmp.lmp.commands_string(f"region gcmc_region_{1} block {gcmc_xlo} {gcmc_xhi} {gcmc_ylo} {gcmc_yhi} {gcmc_zlo} {gcmc_zhi} side in")

        pre_equilibration_lammps_commands = self.equilibriation_thermostat_command(
            initial_temperature=initial_temperature, langevin_damp=langevin_damp, langevin_seed=langevin_seed, fix_gcmc=True
        )
        pre_equilibration_lammps_commands.append(self.equilibriation_chemostat_command(
            n_region=1, N=N_gcmc_1, X=X_gcmc_1, typem=2, seedm=langevin_seed, Tm=1.0, mu=mu_gcmc_1, maxp=max_gcmc_1
        ))

        trilmp.lmp.commands_string('\n'.join([pre_equilibration_lammps_commands]))

        postequilibration_lammps_commands = []
        # Unfix stuff for some reason if in equilibriation
        postequilibration_lammps_commands.append(f"unfix vertexnve")
        postequilibration_lammps_commands.append(f"unfix lvt")


        postequilibration_lammps_commands.append(self.langevin_commands(
            membrane_vertex_mass=membrane_vertex_mass, initial_temperature=initial_temperature,
            langevin_damp=langevin_damp, langevin_seed=langevin_seed
        ))
        trilmp.run(total_sim_time, fix_symbionts_near=False, integrators_defined=True, postequilibration_lammps_commands=postequilibration_lammps_commands)

def main():
    out = SimulationManager(resolution=3).run()

if __name__ == '__main__':
    main()