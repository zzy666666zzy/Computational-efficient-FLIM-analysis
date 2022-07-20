# Computational-efficient-FLIM-analysis

This repository introduced a hardware-friendly 1-D deep learning neural network to analyze average fluorescence lifetime from mono- and bi-exponential decays. And a traing aware quantization was employed to make it more compact. The core modification is using adder-based convolution to make it hardware-friendly. Both synthetic and real data outperform most existing iterative, non-iterative, and 1-D CNN.
 
Figures below are GUI designed based on the neural network, showing phasor plot, FLIM images of synthetic and real data.  
![8b5e9a8b9fd2b33eadfed03adf71fbf](https://user-images.githubusercontent.com/35866553/158862457-3729c52d-cbf1-41f1-9694-45fdb20e0999.jpg)
![image](https://user-images.githubusercontent.com/35866553/160491286-af61dea6-e418-4c23-aa20-ee7433dc1fc7.png)

For more details about the implementation, please refer to the paper. If you find this work useful, please consider citing this article. A hardware version is being developed, stay tuned. 

Citation:
Zang Z, Xiao D, Quan W, Zinuo Li, Chen Y, & Li DDU, "Hardware Inspired Neural Network for Efficient Time-Resolved Biomedical Imaging," 44th IEEE EMBC 2022.
