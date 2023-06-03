# Brain-MRI-Segmentation

Various 2D segmentation algorithms are applied to each slice of the MRI data. The segmented results are compared with the true labels using suitable evaluation metrics. A 3D segmentation algorithm is used to segment the entire MRI data simultaneously and evaluated using the metric used for comparing 2D segmentation.

Otsu's thresholding method and Gaussian Mixture Model-based Hidden Markov Random Field algorithms were implemented using Matlab to achieve Dice coefficient scores of 0.75 and 0.80 in 2-D and 3-D settings respectively.

Link to Technical Report: https://github.com/JatinArutla/Brain-MRI-Segmentation/blob/main/Technical_Report_Jatin.pdf
