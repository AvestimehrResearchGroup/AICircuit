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

simulator = Simulator(circuit_path, circuit_path_docker, circuit_params, params_path, 'oceanScript.ocn')
simulator.run_all(n=args.npoints)
results = simulator.sim_results

if args.circuit == 'Receiver':
   simulator_individual = Simulator(circuit_path, circuit_path_docker, circuit_params, params_path, 'oceanScriptIndividual.ocn')
   simulator_individual.run_all(n=args.npoints)   
   for i in range(len(results)):
      results[i].update(simulator_individual.sim_results[i])
elif args.circuit == 'Mixer':
   simulator_voltage_swing = Simulator(circuit_path, circuit_path_docker, circuit_params, params_path, 'oceanScriptVoltageSwing.ocn')
   simulator_voltage_swing.run_all(n=args.npoints)   
   for i in range(len(results)):
      results[i].update(simulator_voltage_swing.sim_results[i])


# print simulation results
# print(simulator.sim_results)
# result.calc_hist(simulator.sim_results)
# for item in simulator.sim_results:
#    print(item)

p = make_save_path(f'Dataset/{args.circuit}', args.model)
with open(join(p, 'sim_results.csv'),'w', newline='') as f:
   w = csv.DictWriter(f, results[-1].keys())
   w.writeheader()
   w.writerows(results)
   f.close()
