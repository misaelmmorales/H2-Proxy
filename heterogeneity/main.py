################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *
clear_session()
check_tensorflow_gpu()

het = H2_heterogeneity()
het.return_data = True
het.verbose     = True

facies, fluvial_perm, gaussian_perm = het.create_data()
heterogeneity = het.process_perm_poro()
het.plot_samples()