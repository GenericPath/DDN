from subprocess import Popen # to create the python process(es)
import os, time, argparse, shutil # setup of args, folders and files
from itertools import product # for automatically creating all the combinations
import pprint  # for printing/writing a dict nicely
from typing import Iterable # for flattening

def str_flatten(items):
    """Yield string items from any nested iterable; see https://stackoverflow.com/a/40857703"""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from str_flatten(x) # python3 only
        else:
            yield str(x) # or could return the items as original type without str()

parser = argparse.ArgumentParser(description='Runs a series of models with set configurations')
parser.add_argument('--production', action='store_true',
                        help='Production mode: If true run in a separate folder on a copy of the python scripts')

args, unknown = parser.parse_known_args() # unknown args are discarded, useful if running with debugger..

test = '' # TODO: switch to --test when testing
# Defaults (non-production)
script = 'main.py'
out_file = 'runs.txt'

custom = "" # NOTE: used to create a unique folder name in production mode

if args.production:
    # Store the run (and scripts etc) in a month/day/ folder
    folder = 'experiments/' + time.strftime("%m_%d_%H_%M/") + custom

    out_file = folder + out_file
    if not os.path.exists(folder):
        os.makedirs(folder)

    files_to_copy = ['run_model.py', script, 'model.py', 'model_loops.py', 'nc.py', 'data.py', 'node.py', 'net_argparser.py']

    for file in files_to_copy:
        shutil.copy2(file, folder)
    script = folder + script

commands = {
    '-e' : [150],
    '-b' : [32],
    '-lr' : [0.01],
    '-ti' : [5000],
    '-gpu' : [1],
    '--production' : [args.production],
    '--network' : [0],
    '--dataset' : ['simple01'],
    '--radius' : [5,20],
    '--minify' : [True, False],
    '--img-size' : [[16,16]],
    "-m" : [0.9],
    "-v" : [0.1],
    "-s" : [0],
    "--net-size-weights" : [[1,4,8,4],[1,8,16,8]],
    "--net-size-post" : [[1,4,8,4],[1,8,16,8]],
    "--gamma" : [0, 0.5, 1],
    "--eps" : [1]
}

# [dict(zip(d, v)) for v in product(*d.values())]
# from https://stackoverflow.com/a/15211805
runs = [dict(zip(commands, v)) for v in product(*commands.values())]
changing_keys = [key for key, values in commands.items() if len(values) > 1]
print(f'{len(runs)} runs')
print(f'{len(changing_keys)} variables')

# TODO : add profiling (torch.profiler) from pip install pytorch_tb_profiler or w/e

# Run each experiment (not in parallel)
for i, run in enumerate(runs):
    run_name = 'run' + str(i)
    for changing_key in changing_keys: # add all the changing variables to the run name
        run_name += f'{changing_key}-{str(run[changing_key])}'
    run_name += 'custom'
    run['-n'] = run_name

    command = list(sum(run.items(), ()))
    flat = list(str_flatten(command))
    flat.insert(0, script)
    flat.insert(0, 'python')

    pprint.pprint(run)
    f = open(out_file, "a")
    f.write(run_name + '\n')
    f.write(pprint.pformat(run))
    f.close()
    p = Popen(flat)
    (output, err) = p.communicate()