# Simulation class given analog circuit parameters
#
# Author: Yue (Julien) Niu

import csv
import subprocess


# DOCKER_CMD = 'docker exec --user=julien rlinux8-1 /bin/tcsh -c -v /home/asal/AICircuit/Simulation:./'
DOCKER_CMD = 'docker exec --user=asal rlinux8-2 /bin/tcsh -c'
# OCEAN_FILENAME = 'oceanScript.ocn'


class Simulator:
    
    def __init__(self, circuit_path, circuit_path_docker, circuit_params, params_path, ocean_filename):
        """
        :param circuit_path: circuit path in a host system
        :param circuit_path_docker: circuit path inside a docker container
        :param circuit_params: defined circuit parameters
        :param params_path: circuit parameter value path
        """
        self.circuit_path = circuit_path
        self.circuit_def = circuit_path + '/' + ocean_filename
        self.circuit_params = circuit_params
        self.params_path = params_path
        
        # construct simulation command with docker
        self.circuit_path_docker = circuit_path_docker
        self.docker_cmd = DOCKER_CMD
        
        # store simulation results
        self.sim_results = []


    def run_sim(self):
        """Start a simulation
        Note that all simulation parameters are defined in input.scs file.
        If additional simulation functions need to be added, you should directly edit input.scs file.
        """
        
        bash_cmds = f'\"cd {self.circuit_path_docker}; ocean -nograph -replay oceanScriptNew.ocn"'
        sim_cmd = f'{self.docker_cmd} {bash_cmds}'
        # raise ValueError(sim_cmd, bash_cmds)
        # print(sim_cmd)
        ret = subprocess.call(sim_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        if ret:
            print('[ERROR] cmd is not properly executed!!!')

                
    def get_results(self):
        """Simply extract result from results.txt generated from ocean
        """
        cur_sim_result = {}
        result_path = self.circuit_path + '/results.txt'
        result_file = open(result_path, 'r')
        lines = result_file.readlines()
        for line in lines:
            if ':' in line:
                try:
                    metric, value = line.split(':')[0], float(line.split(':')[1])
                except ValueError:
                    return
                else:
                    cur_sim_result[metric] = value
                    cur_sim_result['Error_'+metric] = 0
                    cur_sim_result[metric + '_GroundTruth'] = 0
                    
        self.sim_results.append(cur_sim_result)


    def run_all(self, n=10, display=True):
        """Run all simulations by sweeping paramters defined in the .csv file
        """
        with open(self.params_path, mode='r') as param_file:
            param_dict = csv.DictReader(param_file)
            for i, line in enumerate(param_dict):
                for p in self.circuit_params:
                    if p not in line: continue
                    
                    self.circuit_params[p] = line[p]
            
                # edit circuit parameters in .scs file
                alter_circ_param(self.circuit_params, self.circuit_def)

                # start simulation
                self.run_sim()
                
                # get simulation results
                self.get_results()

                # calculate relative error
                self.calc_error(line)

                if display and i > 10 and (i+1) % 50 == 0:
                    print('{} points simulated.'.format(i+1))

                if n != -1 and i == n - 1: break


    def calc_error(self, perf_ref):
        """Calculate error compared to reference values
        :param perf_ref: reference performance
        """
        for key in self.sim_results[-1]:
            if 'Error' not in key and 'GroundTruth' not in key:  # only check actual values, not error
                val_actual = self.sim_results[-1][key]
                val_ref = float(perf_ref[key])
                val_ref_save = val_ref

                if 'VoltageGain' in key:
                    val_actual = 10 ** (val_actual / 20)
                    val_ref = 10 ** (val_ref / 20)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                elif 'ConversionGain' in key or 'PowerGain' in key or 'NoiseFigure' in key or 'S11' in key or 'S22' in key:
                    val_actual = 10 ** (val_actual / 10)
                    val_ref = 10 ** (val_ref / 10)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                else:
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)

                self.sim_results[-1]['Error_' + key] = rel_error
                self.sim_results[-1][key + '_GroundTruth'] = val_ref_save


def alter_circ_param(new_params_values, ocean_path):
    scs_file = open(ocean_path, 'r')
    lines = scs_file.readlines()
    
    # format for set variable values
    format_var = 'desVar(   \"{}\" {} )\n'

    # locate the line of circuit parameters
    for i, line in enumerate(lines):
        if 'desVar' in line:
            var = line.split('\"')[1]
            if var in new_params_values:
                if new_params_values[var] == 0: continue
                
                lines[i] = format_var.format(var, new_params_values[var])

    ocean_path_new = '/'.join(ocean_path.split('/')[0:-1]) + '/oceanScriptNew.ocn'
    ocean_file_new = open(ocean_path_new, 'w')
    ocean_file_new.writelines(lines)