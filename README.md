# H2-Proxy
proxy modeling for subsurface H2 storage


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
