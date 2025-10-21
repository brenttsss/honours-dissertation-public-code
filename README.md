# Honour's Dissertation

This repository contains the code used for "Extracting the $W\to\mu$ Signal in the Forward Rapidity Region of $pp$ Collisions in ALICE Run 3 Data using Machine-Learning Techniques."

The code contains four classes. The class ``DecisionTree`` contains the global settings for all classifier algorithms, as well as the plotting functions. The classes ``ModelXGB``, ``ModelLGBM`` and ``ModelLR`` inherits the properties of ``DecisionTree``.

The class ``DecisionTree`` contains methods for plotting the score histograms, confusion matrix, Reciever Operating Characteristic curve, Precision-Recall curve and Feature Importance.

The classes ``ModelXGB``, ``ModelLGBM`` and ``ModelLR`` contains the classifier methods and helper functions that call the methods in ``DecisionTree`` to plot the results from the internal methods.
