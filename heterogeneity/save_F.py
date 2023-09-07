import numpy as np
import os, glob, math
from scipy.io import savemat

nx, ny, nz = 256, 256, 1
my_dir   = os.getcwd()
data_dir = '/project/mfr/shaowen/7.HETE_2D_Fluvial_VaryGeometry/samples_F/RESULTS/FUNRST'

os.chdir(data_dir)
directories = next(os.walk(data_dir))[1]
keys = ['SGAS', 'YMF_3', 'PERMX', 'PORO', 'PRESSURE']

for directory in directories:
    os.chdir(os.path.join(data_dir,directory))
    funrst_files = glob.glob('*.FUNRST')
    for file in funrst_files:
        arrays_dict = {}
        with open(file, 'r') as f:
            lines = iter(f)
            line = next(lines, None)
            while line:
                if f"{nx*ny*nz} 'REAL'" in line:
                    key = line[2:10].strip()
                    if key in keys:
                        data = []
                        for _ in range(math.ceil(nx*ny*nz/4)):
                            line = next(lines, None)
                            if line:
                                data.extend(line.strip().split())
                        array = np.array(data, dtype=float).reshape((nx, ny, nz), order='F')
                        arrays_dict[key] = array
                line = next(lines, None)
        savemat('/project/MFR2/misael/h2dataf/{}.mat'.format(os.path.splitext(file)[0]), arrays_dict)
    os.chdir('../')
os.chdir(my_dir)