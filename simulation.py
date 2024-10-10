# This is the top entry of the codebase

from Simulation.simulator import Simulator
from Simulation.utils.param import get_circ_params, get_circ_path, get_dataset_path
from Simulation.utils import result
from Simulation.args import args
import csv
from Utils.utils import make_save_path
from os.path import join


circuit_params = get_circ_params(args.circuit)
circuit_path, circuit_path_docker = get_circ_path(args.circuit)
params_path = get_dataset_path(args.circuit, args.model)

simulator = Simulator(circuit_path, circuit_path_docker, circuit_params, params_path)

simulator.run_all(n=args.npoints)

# print simulation results

# print(simulator.sim_results)

# result.calc_hist(simulator.sim_results)
# for item in simulator.sim_results:
#    print(item)

p = make_save_path(f'Dataset/{args.circuit}', args.model)
with open(join(p, 'sim_results.csv'),'w', newline='') as f:
   w = csv.DictWriter(f, simulator.sim_results[-1].keys())
   w.writeheader()
   w.writerows(simulator.sim_results)
   f.close()
