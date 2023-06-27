<<<<<<< Updated upstream
################################################################
########################    H2 Proxy    ########################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *
h2 = H2Toolkit()
h2.check_torch_gpu()
h2.return_plot = True

########################## PROCESSING ##########################
h2.read_data()
h2.process_data(restype='SA')
#h2.load_data()

############################# ROM ##############################
rom  = h2_cushion_rom(h2.inp, h2.out, hidden_sizes=[32,64,128,64,32])
opt  = optim.NAdam(rom.parameters(), lr=1e-3, weight_decay=1e-7)
loss = L1L2_Loss()
loss.l2_weight = 0.25

h2.train(rom, loss, opt)
h2.plot_loss()

######################## PLOTS & PRINTS ########################
h2.make_predictions(rom)
h2.print_metrics()
h2.plot_results()

################################################################
############################## END #############################
=======
################################################################
########################    H2 Proxy    ########################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *
h2 = H2Toolkit()
h2.check_torch_gpu()
h2.return_plot = True

########################## PROCESSING ##########################
h2.read_data()
h2.process_data(restype='SA')
#h2.load_data()

############################# ROM ##############################
rom  = h2_cushion_rom(h2.inp, h2.out, hidden_sizes=[32,64,128,64,32])
opt  = optim.NAdam(rom.parameters(), lr=1e-3, weight_decay=1e-6)
loss = L1L2_Loss()
loss.l2_weight = 0.2

h2.train(rom, loss, opt)
h2.plot_loss()

######################## PLOTS & PRINTS ########################
h2.make_predictions(rom)
h2.print_metrics()
h2.plot_results()

################################################################
############################## END #############################
>>>>>>> Stashed changes
################################################################