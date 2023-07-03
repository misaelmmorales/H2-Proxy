import os, glob, math
import numpy as np

## Global variables
my_dir           = 'C:/Users/381792/Documents/H2-Proxy/heterogeneity/'
dgr_fluvial_dir  = '//dcstorage.lanl.gov/mfr/misael/depleted_gas_reservoir/2D_Fluvial'
dgr_gaussian_dir = '//dcstorage.lanl.gov/mfr/misael/depleted_gas_reservoir/2D_Gaussian/'
sa_fluvial_dir   = '//dcstorage.lanl.gov/mfr/misael/saliine_aquifer/2D_Fluvial'
sa_gaussian_dir  = '//dcstorage.lanl.gov/mfr/misael/saliine_aquifer/2D_Gaussian/'
nx, ny, nz = 256, 256, 1

## Choose a target directory
choose = 1

if choose==1:
    target_dir = dgr_fluvial_dir
elif choose==2:
    target_dir = dgr_gaussian_dir
elif choose==3:
    target_dir = sa_fluvial_dir
elif choose==4:
    target_dir = sa_gaussian_dir
else:
    print('please choose a target directory')


## save loop
os.chdir(target_dir)
directories = next(os.walk('.'))[1]
for directory in directories:
    if directory == 'FUNRST':
        continue
    os.chdir(directory)
    funrst_files = glob.glob('*.FUNRST')
    for file in funrst_files:
        arrays_dict = {}
        with open(file, 'r') as f:
            lines = iter(f)
            line = next(lines, None)
            while line:
                if f"{nx*ny*nz} 'REAL'" in line:
                    key = line[2:10].strip()
                    data = []
                    for _ in range(math.ceil(nx*ny*nz/4)):
                        line = next(lines, None)
                        if line:
                            data.extend(line.strip().split())
                    array = np.array(data, dtype=float).reshape((nx, ny, nz), order='F')
                    arrays_dict[key] = array
                line = next(lines, None)
        for key, array in arrays_dict.items():
            filename = os.path.splitext(file)[0]
            if not os.path.exists(filename):
                os.makedirs(filename)
            np.save(os.path.join(filename, key), array)
    os.chdir('../')