import sys
import os
import time
from subprocess import Popen, PIPE, STDOUT
from analysis_and_visualization import retrieve_result_dirs

core_number = 1
path_to_mozaik_env = '/home/rozsa/virt_env/mozaik_lgn/bin/activate'
exclude = "--exclude=w[14-17]"

assert len(sys.argv) == 2
directory = sys.argv[1]

for path in retrieve_result_dirs(directory):
    p = Popen(['sbatch'] +  ['-o',directory+"/slurm_analysis-%j.out" ],stdin=PIPE,stdout=PIPE,stderr=PIPE,text=True)
    data = '\n'.join([
                        '#!/bin/bash',
                        '#SBATCH -J MozaikParamSearchAnalysis',
                        '#SBATCH -c ' + str(core_number),
                        '#SBATCH %s' % exclude,
                        'source %s' % path_to_mozaik_env,
                        'cd ' + os.getcwd(),
                        'echo "Running analysis through slurm"',
                        ' '.join(["python run_analysis_and_visualization.py","'"+path+"'"]  +['>']  + ["'"+path +'/OUTFILE_analysis'+str(time.time()) + "'"]),
                    ])
    print(p.communicate(input=data)[0])
    print(data)
    p.stdin.close()
