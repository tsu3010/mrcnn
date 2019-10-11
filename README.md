# Mark R CNN for Ship detection

This package is an implementation of Mask R CNN (Mask Region based Convolution Neural Networks) algorithm implemented by [Gabriel Garza](https://github.com/gabrielgarza/Mask_RCNN). This model is implemented for detection of ships from satellite images. The implementation by Gabriel Garza has a few missing components, which are added in this package. More details on the package can be found here <https://towardsdatascience.com/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083> 

## Introduction to Mask R CNN

Mask R CNN is an extension of Faster R CNN. Faster R-CNN predicts bounding boxes and Mask R-CNN essentially adds one more branch for predicting an object mask in parallel. MAsk R CNN follows the following general steps:
1. __Backbone model__: a standard convolutional neural network that serves as a feature extractor. For example, it will turn a1024x1024x3 image into a 32x32x2048 feature map that serves as input for the next layers.
2. __Region Proposal Network (RPN)__: Using regions defined with as many as 200K anchor boxes, the RPN scans each region and predicts whether or not an object is present. One of the great advantages of the RPN is that does not scan the actual image, the network scans the feature map, making it much faster.
3. __Region of Interest Classification and Bounding Box__: In this step the algorithm takes the regions of interest proposed by the RPN as inputs and outputs a classification (softmax) and a bounding box (regressor).
4. __Segmentation Masks__: In the final step, the algorithm the positive ROI regions are taken in as inputs and 28x28 pixel masks with float values are generated as outputs for the objects. During inference, these masks are scaled up.

more information on MAsk R CNN can be found in following resources
- <https://medium.com/free-code-camp/mask-r-cnn-explained-7f82bec890e3>

## Shortfalls

Apart from solving the AI problem, any machine learning solution should robust like an industrial product. It should be testable, scalable, recoverable, etc. If an application has achieve a fair level of these commandments, it can be considered a dependable solution, and can be extended to solve other ML problems. More details on this [page](https://www.linkedin.com/pulse/eleven-commandments-framework-building-machine-learning-soma-dhavala)

## Additions in the package


### Build process

The package is build using pybuilder. The build process has the following features:
1. __Module testing__: Each function of the package is tested using `pytest`. The package will not build if any test fails. This feature improves code testability
2. __Coverage test__: The build looks for a 70% coverage of unittests. This enforces the author to write test cases for all modules, classes and functions in the core library.
3. __Documentation__: Auto-generated documentation is generated using the sphinx library. Documentation can be configured by `mrcnn/docs/source/conf.py` file. This improves code readibility and diagnosibility
4. __Static versions__: Cross-version compatibility of dependent libraries are critical to a fail-safe package. Therefore, version of all the dependent libraries are fixed in `mrcnn/requirements.txt`. This imporves robustness, repeatibility and reproducability. 


## Scripts

1. __Jobs__: Jobs such as training ship data, training shapes data, running coco model are scripted in `jobs/` folder, and are called upon in the `mrcnn/src/main/scripts/mrcnn` file. The functions used in this script can be accessed from cmd, once the build package is installed from the `.tar` file. This makes code delivery simpler and user friendly

2. __Job Arguments__: APIs are exposed to take in job arguments and the arguments are validated in each script.

## Next Steps

1. Improve parallelization using Kubernetes or EMR clusters. This will improve scalability and robusness of the module.
2. Improve the code testing and add more tests in `mrcnn/src/unittest/python/mrcnn`
3. Add more jobs in `mrcnn/src/main/scripts/` and `mrcnn/src/main/python/mrcnn/python`
4. Execute end-to-end pipeline
