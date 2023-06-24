################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *

hete = Heterogeneity()
facies = hete.load_facies()
hete_fluv, hete_gaus = hete.load_perm_poro()