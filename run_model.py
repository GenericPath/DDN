from subprocess import Popen
# name generated by for loop...
epochs = [50]
batch_sizes = [2]
lrs = [1e-2]
momentums = [0.9]
vals = [0.1]
seeds = [0]
total_imageses = [10, 300, 5000]
net_sizes = [[1,128,256,512,1024]]


i=0
    
for epoch in epochs:
    for batch_size in batch_sizes:
        for lr in lrs:
            for total_images in total_imageses:
                for net_size in net_sizes:
                    if total_images == 5000:
                        batch_size = 1 # just to ensure memory is good
                    i += 1
                    run_name = 'slowerAbs' + str(i)
                    command = ["python", "model.py", 
                                "-n", str(run_name),
                                "-e", str(epoch),
                                "-b", str(batch_size),
                                "-lr", str(lr),
                                "-ti", str(total_images),
                                "-ns", str(net_size[0]), str(net_size[1]), str(net_size[2]),  str(net_size[3]), str(net_size[4]),
                                "-gpu", str(1)]
                                # "-m", momentum,
                                # "-v", val,
                                # "-s", seed,
                                # ""]
                    print(command)
                    p = Popen(command)
                    (output, err) = p.communicate()

# Was also run on larger models, except they were all the 300 model.
# also these may have been run on AbstractDeclarativeNode instead of EqConstrainedDeclarativeNode
# ['python', 'model.py', '-n', 'run3', '-e', '30', '-b', '1', '-lr', '1e-08', '-ti', '300', '-ns', '1', '128', '256', '512', '1024']
# ['python', 'model.py', '-n', 'run4', '-e', '30', '-b', '1', '-lr', '1e-08', '-ti', '300', '-ns', '1', '16', '32', '64', '1024']
# ['python', 'model.py', '-n', 'run9', '-e', '30', '-b', '1', '-lr', '0.0001', '-ti', '300', '-ns', '1', '128', '256', '512', '1024']
# ['python', 'model.py', '-n', 'run10', '-e', '30', '-b', '1', '-lr', '0.0001', '-ti', '300', '-ns', '1', '16', '32', '64', '1024']

# addition of LrReduceOnPlateau
# REMEBER: quicks is in the main folder, slowers is in DDN2/DDN.. as otherwise the main could would've been overwritten with differences between them
# quicks (EqConstr).. (lr reduce on plateau for patience = 10)
# ['python', 'model.py', '-n', 'quick1', '-e', '50', '-b', '1', '-lr', '1e-05', '-ti', '50', '-ns', '1', '32', '32', '64', '1024', '-gpu', '0']
# ['python', 'model.py', '-n', 'quick2', '-e', '50', '-b', '1', '-lr', '0.0001', '-ti', '50', '-ns', '1', '32', '32', '64', '1024', '-gpu', '0']
# ['python', 'model.py', '-n', 'quick3', '-e', '50', '-b', '1', '-lr', '0.01', '-ti', '50', '-ns', '1', '32', '32', '64', '1024', '-gpu', '0']
# slowers (Abstract)... (lr reduce on plateau for patience = 3)
# ['python', 'model.py', '-n', 'slowerAbs1', '-e', '50', '-b', '2', '-lr', '0.01', '-ti', '10', '-ns', '1', '128', '256', '512', '1024', '-gpu', '1']
# ['python', 'model.py', '-n', 'slowerAbs2', '-e', '50', '-b', '2', '-lr', '0.01', '-ti', '300', '-ns', '1', '128', '256', '512', '1024', '-gpu', '1']
# ['python', 'model.py', '-n', 'slowerAbs3', '-e', '50', '-b', '1', '-lr', '0.01', '-ti', '5000', '-ns', '1', '128', '256', '512', '1024', '-gpu', '1']