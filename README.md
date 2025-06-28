# H2-Proxy

Mao, S., Chen, B., Morales, M. M., Malki, M., and Mehana, M. (2024). Cushion gas effects on hydrogen storage in porous rocks: Insights from reservoir simulation and deep learning. <em>Hydrogen Energy</em>. https://doi.org/10.1016/j.ijhydene.2024.04.288.

<p align="center">
  <img src="https://github.com/misaelmmorales/H2-Proxy/blob/main/cushion/cushion.jpg" width="1000"/>
</p>


<!-- 
proxy modeling for subsurface H2 storage

256x256x1 SGEMS params: 
(Max, Med, Min) = (100, 50, 25)
(Nugget, Contrib) = (0.01, 0.3)

To-Do:
1) Create (500) 256x256x1 {Kx, $\phi$} Gaussian Fields -> kx from SGEMS, $\phi$ from KZ
2) Create (500) 256x256x1 {Kx, $\phi$} Fluvial fields -> from [MLTrainingImages](https://github.com/misaelmmorales/MLTrainingImages) (slice 2D)
3) Upload individual realizations, col=idx | row=jdx

TensorFlow GPU setup:
Go into Anaconda Navigator and edit condarc settings to include http and https proxy setting
- conda deactivate
- conda create -n deep
- conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
- pip install --proxy http://proxyout.lanl.gov:8080 tensorflow

https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step/
https://medium.com/swlh/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805
https://appsilon.com/visualize-pytorch-neural-networks/
-->
