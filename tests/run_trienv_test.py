import numpy as np

from util import icosphere
from trimem.mc.trilmp import TriLmp, Beads
from trimesh import Trimesh

class BasePairStyle():
    def __init__(self, name: str, coeff_commands: list[str] = [], modify_commands: list[str] = []):
        self.name = name
        self.coeff_commands = coeff_commands
        self.modify_commands = modify_commands

    @property
    def init_params(self) -> list:
        return []

    def get_init_string(self) -> str:
        return ' '.join([self.name]+self.init_params)
    
    def parse_coeff_commands(self):
        return ['pair_coeff '+ coeff_command for coeff_command in self.coeff_commands]
    
    def parse_modify_commands(self):
        return ['pair_modify '+ modify_command for modify_command in self.modify_commands]

class TablePairStyle(BasePairStyle):
    def __init__(self, style: str = 'linear', N: int = 2000, *args, **kwargs):
        super().__init__(name='table', *args, **kwargs)
        self.style = style
        self.N = N

    @property
    def init_params(self) -> list:
        return [self.style, str(self.N)]
    
class LJCutPairStyle(BasePairStyle):
    def __init__(self, cutoff: float=2.5, *args, **kwargs):
        super().__init__(name='lj/cut', *args, **kwargs)
        self.cutoff = cutoff

    @property
    def init_params(self) -> list:
        return [str(self.cutoff)]
    
    def set_membrane_attraction(self, n_types, interaction_strength, sigma_tilde, interaction_range):
        self.coeff_commands = ['* * lj/cut 0 0 0']
        for i_regular in range(1, n_types):
            self.coeff_commands.append(f'1 {i_regular+1} lj/cut {interaction_strength} {sigma_tilde} {interaction_range*sigma_tilde}')
        
        self.modify_commands = ['pair lj/cut shift yes']

class HarmonicCutPairStyle(BasePairStyle):
    def __init__(self, *args, **kwargs):
        super().__init__(name='harmonic/cut', *args, **kwargs)

    def set_metabolite_repulsive_commands(self, n_types, sigma_metabolites):
        self.coeff_commands = ['* * harmonic/cut 0 0']
        # Add interactions between regular particles
        for i_regular in range(1, n_types):
            for j_regular in range(i_regular, n_types):
                self.coeff_commands.append(f"{i_regular+1} {j_regular+1} harmonic/cut 1000 {sigma_metabolites}")

    def set_all_repulsive_commands(self, n_types, sigma_metabolites, sigma_special):
        self.set_metabolite_repulsive_commands(n_types=n_types, sigma_metabolites=sigma_metabolites)

        # Add interactions with special sphere
        for i_regular in range(1, n_types):
            self.coeff_commands.append(f"1 {i_regular+1} harmonic/cut 1000 {sigma_special}")

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

        self.pair_styles: list[BasePairStyle] = []

    def init_trilmp(self):
        # membrane parameters
        self.membrane_vertex_mass = 1.0
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
        self.total_sim_time=1000 # 00  # in time units
        discrete_snapshots=10   # in time units
        print_frequency = int(discrete_snapshots/(step_size*traj_steps))

        self.initial_temperature=1.0                    # MD PART SIMULATION: temperature of the system
        pure_MD=False,                             # MD PART SIMULATION: accept every MD trajectory?
        self.langevin_damp=1.0
        self.langevin_seed=123
        (xlo,xhi,ylo,yhi,zlo,zhi) = (-50, 50, -50, 50, -50, 50)
        switch_mode = 'random'

        # Interaction parameters
        self.interaction_range = 1.5 # rc_mm
        self.interaction_strength = 10

        # GCMC region
        variable_factor = 10
        self.N_gcmc_1=int(variable_factor*self.langevin_damp/step_size)
        self.X_gcmc_1=100
        self.seed_gcmc_1 = self.langevin_seed
        self.mu_gcmc_1=0
        vfrac = 0.01
        geometric_factor = 1.0
        self.sigma_metabolites = 1.0
        self.sigma_tilde_membrane_metabolites = 0.5*(self.sigma_metabolites+self.sigma_membrane)
        x_membrane_max = np.max(self.mesh.vertices[:, 0])
        r_mean = np.mean(
            np.linalg.norm(
                np.mean(self.mesh.vertices, axis=0)-self.mesh.vertices, axis=1
            )
        )
        height_width = r_mean*geometric_factor
        self.gcmc_xlo, self.gcmc_xhi = x_membrane_max, xhi, # xlo, xhi
        self.gcmc_ylo, self.gcmc_yhi = -height_width/2, height_width/2, # ylo, yhi
        self.gcmc_zlo, self.gcmc_zhi = -height_width/2, height_width/2, # zlo, zhi

        vtotal_region = (xhi-x_membrane_max)*(height_width)*(height_width)
        self.max_gcmc_1 = int((vfrac*vtotal_region*3)/(4*np.pi*(self.sigma_metabolites*0.5)**3))

        self.trilmp = TriLmp(
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
            mass_particle_type=[self.membrane_vertex_mass, 1.],# the mass of the particle per type
            group_particle_type=['vertices', 'metabolites'],

            step_size=step_size,                      # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
            traj_steps=traj_steps,                    # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
            flip_ratio=flip_ratio,                    # MC PART SIMULATION: fraction of edges to flip?
            # check_neigh_every=1,                      # NEIGHBOUR LISTS
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
        return f"fix mygcmc_{n_region} metabolites gcmc {N} {X} 0 {typem} {seedm} {Tm} {mu} 0 region gcmc_region_{n_region} max {maxp}"

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

    @staticmethod
    def walls_command(ylo, yhi, zlo, zhi):
        return f"fix ConfinementMet metabolites wall/reflect xhi EDGE ylo {ylo} yhi {yhi} zlo {zlo} zhi {zhi}"

    def get_pair_style_commands(self):
        # This command tells lamps all the styles that are to be expected
        style_command = f"pair_style hybrid/overlay"
        coeff_and_modify_commands = []
        for pair_style in self.pair_styles:
            # add parameter and initial parameters
            style_command = ' '.join([style_command, pair_style.get_init_string()])
            coeff_and_modify_commands += pair_style.parse_coeff_commands() + pair_style.parse_modify_commands()

        return '\n'.join([style_command]+coeff_and_modify_commands)
            
    def run(self):
        table_ps = TablePairStyle(
            style='linear', N=2000,
            coeff_commands = ['1 1 table trimem_srp.table trimem_srp'],
            modify_commands = ['pair table special lj/coul 0.0 0.0 0.0 tail no']
        )
        harmonic_ps = HarmonicCutPairStyle()
        harmonic_ps.set_all_repulsive_commands(2, self.sigma_membrane, self.sigma_tilde_membrane_metabolites)
        lj_ps = LJCutPairStyle(cutoff=2.5)
        lj_ps.set_membrane_attraction(2, interaction_strength=self.interaction_strength, sigma_tilde=self.sigma_tilde_membrane_metabolites, interaction_range=self.interaction_range)
        self.pair_styles = [table_ps, harmonic_ps]
        
        # add a gcmc region
        self.trilmp.lmp.commands_string(f"region gcmc_region_{1} block {self.gcmc_xlo} {self.gcmc_xhi} {self.gcmc_ylo} {self.gcmc_yhi} {self.gcmc_zlo} {self.gcmc_zhi} side in")

        pre_equilibration_lammps_commands = []
        # Add walls before equilibriation
        pre_equilibration_lammps_commands.append(self.walls_command(
            self.gcmc_ylo, self.gcmc_yhi, self.gcmc_zlo, self.gcmc_zhi
        ))
        # Set equilibriation thermostat
        # TODO: only different because of tally yes, remove?
        pre_equilibration_lammps_commands.append(self.equilibriation_thermostat_command(
            initial_temperature=self.initial_temperature, langevin_damp=self.langevin_damp, langevin_seed=self.langevin_seed, fix_gcmc=True
        ))
        # Set chemostat before equilibriation
        pre_equilibration_lammps_commands.append(self.equilibriation_chemostat_command(
            n_region=1, N=self.N_gcmc_1, X=self.X_gcmc_1, typem=2, seedm=self.langevin_seed, Tm=1.0, mu=self.mu_gcmc_1, maxp=self.max_gcmc_1
        ))
        pre_equilibration_lammps_commands.append(self.get_pair_style_commands())
        
        # Set interaction parameters
        self.trilmp.lmp.commands_string('\n'.join(pre_equilibration_lammps_commands))

        postequilibration_lammps_commands = []
        # Unfix stuff for some reason if in equilibriation
        postequilibration_lammps_commands.append(f"unfix vertexnve")
        postequilibration_lammps_commands.append(f"unfix lvt")


        # Fix thermostat again without tally yes
        postequilibration_lammps_commands.append(self.langevin_commands(
            membrane_vertex_mass=self.membrane_vertex_mass, initial_temperature=self.initial_temperature,
            langevin_damp=self.langevin_damp, langevin_seed=self.langevin_seed
        ))
        # Activate attraction and reset interactions
        harmonic_ps.set_metabolite_repulsive_commands(n_types=2, sigma_metabolites=self.sigma_metabolites)
        self.pair_styles += [lj_ps]
        postequilibration_lammps_commands.append(self.get_pair_style_commands())

        self.trilmp.run(self.total_sim_time, fix_symbionts_near=False, integrators_defined=True, postequilibration_lammps_commands=postequilibration_lammps_commands)

def main():
    smanager = SimulationManager(resolution=3)
    smanager.init_trilmp()
    out = smanager.run()

if __name__ == '__main__':
    main()