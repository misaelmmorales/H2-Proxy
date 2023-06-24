################################################################
########################    H2 Proxy    ########################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *
check_torch_gpu()

h2 = h2proxy()
h2.return_data = True

########################## PROCESSING ##########################
all_data = h2.read_data(n_subsample=5000)
X_train, X_test, y_train, y_test, x_scaler, y_scaler = h2.process_data(restype='SA')

######################## PLOTS & PRINTS ########################
h2.print_metrics()
h2.plot_results()
h2.save_data()

################################################################
############################## END #############################
################################################################


''' OLD TF IMPLEMENTATION:
model, fit = h2.make_model()
if h2.NN_verbose==True:
    h2.model.summary()
h2.plot_loss(figsize=(4,3))
y_train_pred, y_test_pred = h2.make_predictions()
'''