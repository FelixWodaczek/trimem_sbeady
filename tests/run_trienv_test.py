import numpy as np

from util import icosphere
from trimem.mc.trilmp import TriLmp
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
    def nve_command_string(initial_temperature, langevin_damp, langevin_seed, bds_info_lgv=None):
        commands = []
        commands.append("unfix lvt")
        commands.append(f"fix lvt all langevin {initial_temperature} {initial_temperature} {langevin_damp} {langevin_seed} zero yes {bds_info_lgv}")
        return '\n'.join(commands)

    def run(self):
        # membrane parameters
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

        initial_temperature=1.0,                  # MD PART SIMULATION: temperature of the system
        pure_MD=True,                             # MD PART SIMULATION: accept every MD trajectory?
        langevin_damp=1.0
        langevin_seed=123
        total_sim_time = self.param_dict['total_sim_time']
        (xlo,xhi,ylo,yhi,zlo,zhi) = (-50, 50, -50, 50, -50, 50)
        switch_mode = 'random'

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
            
            step_size=step_size,                      # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
            traj_steps=traj_steps,                    # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
            flip_ratio=flip_ratio,                    # MC PART SIMULATION: fraction of edges to flip?

            box=(xlo,xhi,ylo,yhi,zlo,zhi),

            output_prefix='data/data',                # OUTPUT: prefix for output filenames
            restart_prefix='data/data',               # OUTPUT: name for checkpoint files
            checkpoint_every=print_frequency,         # OUTPUT: interval of checkpoints (alternating pickles)
            output_format='lammps_txt',               # OUTPUT: choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
            output_counter=0,                         # OUTPUT: initialize trajectory number in writer class
            performance_increment=print_frequency,    # OUTPUT: output performace stats to prefix_performance.dat file
            energy_increment=print_frequency,         # OUTPUT: output energies to energies.dat file
        )

        postequilibration_lammps_commands = []
        postequilibration_lammps_commands.append(self.nve_command_string(initial_temperature=initial_temperature, langevin_damp=langevin_damp, langevin_seed=langevin_seed, bds_info_lgv='None'))

        trilmp.run(total_sim_time, fix_symbionts_near=False, integrators_defined=True, postequilibration_lammps_commands=postequilibration_lammps_commands)

def main():
    out = SimulationManager().run()

if __name__ == '__main__':
    main()