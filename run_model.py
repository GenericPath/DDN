from subprocess import Popen
import os, time, argparse, shutil
from itertools import product # for automatically creating all the combinations
import pprint  # for printing/writing a dict nicely

parser = argparse.ArgumentParser(description='Runs a series of models with set configurations')
parser.add_argument('--production', action='store_true',
                        help='Production mode: If true run in a separate folder on a copy of the python scripts')

args, unknown = parser.parse_known_args() # unknown args are discarded, useful if running with debugger..

# TODO: add test as an option here?
test = '' # Switch to --test when testing

script = 'main.py'
folder = ""
out_file = 'runs.txt'

if args.production:
    # Store the run (and scripts etc) in a month/day/ folder
    folder = 'experiments/' + time.strftime("%m_%d_%H_%M/")
    out_file = folder + out_file
    if not os.path.exists(folder):
        os.makedirs(folder)

    files_to_copy = ['run_model.py', script, 'model.py', 'model_loops.py', 'nc.py', 'data.py', 'node.py', 'net_argparser.py']

    for file in files_to_copy:
        shutil.copy2(file, folder)
    script = folder + script

commands = {
    '-e' : [10],
    '-b' : [1, 32, 64],
    '-lr' : [0.01],
    '-ti' : [5000],
    '-gpu' : [1],
    '--production' : [args.production],
    '--network' : [0,1],
    '--dataset' : ['simple01'],
    '--radius' : [1,5,10],
    '--minify' : [True, False],
    '--img-size' : [[16,16]],
    "-m" : [0.9, 0.95, 0.8],
    "-v" : [0.1],
    "-s" : [0],
    "--net-size-weights" : [[1,4,8,4]],
    "--net-size-post" : [[1,4,8,4]],
} # append a -n to each permutation

# [dict(zip(d, v)) for v in product(*d.values())]
# from https://stackoverflow.com/a/15211805
runs = [dict(zip(commands, v)) for v in product(*commands.values())]
print(f'{len(runs)} runs')
changing_keys = [key for key, values in commands.items() if len(values) > 1]
print(f'{len(changing_keys)} changing variables')


# TODO : add profiling (torch.profiler) from pip install pytorch_tb_profiler or w/e

i=0
for run in runs:
    run_name = 'run' + str(i)
    for changing_key in changing_keys: # add all the changing variables to the run name
        run_name += f'{changing_key}-{str(run[changing_key])}-'
    # run_name += 'custom'
    run['-n'] = run_name

    # TODO: convert to list
    # TODO: add 'python', script to the start of the list

    pprint.pprint(run)
    f = open(out_file, "a")
    f.write(run_name + '\n')
    f.write(pprint.pformat(run))
    f.close()
    p = Popen(run)
    (output, err) = p.communicate()