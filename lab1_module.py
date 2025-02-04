#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:10:08 2024

Lab Module 1 - 1 function

This module has a function that creates a historgram given an array of numerical values (features) and an array of strings with two mutually
exclusive values (labels). It will also return two arrays with the feature data sorted by the two label categories. 

Sources: None

@author: Lauren Downs and Kristof Rohaly-Medved
"""

#%% Imported Modules
from matplotlib import pyplot as plt

def plot_histogram(features, labels):
    """
    Creates a historgram given an array of numerical values (features) and an array of strings(labels), which can be either 'yes' or another string, and returns two arrays with the feature data sorted by the two label categories. 

    Parameters
    ----------
    features : numpy array (1D) of floats or ints
        Numerical data for each sample
    labels : numpy array (1D) of strs
        Strings equal to either 'yes' or any other string not equal to 'yes' for each sample

    Returns
    -------
    family_history_obese_weights : numpy array of floats or ints
        All of the samples from the `features array that had 'yes' in the `labels arra. This will have the same object type as the `features data.
    non_family_history_weights : numpy array of floats or ints
        All of the samples from the `features array that did not have 'yes' in the `labels array. This will have the same object type as the `features data.

    """
    
    # Create boolean index for label
    is_family_obese = (labels == 'yes')

    # Split weight into obese family and not obese family history
    family_history_obese_weights = features[is_family_obese]
    non_family_history_weights = features[~is_family_obese]
    
    # Plotting
    plt.hist(family_history_obese_weights, edgecolor='black', linewidth=1.5, alpha = 0.3, label='Family History\n of Obesity Weight')
    plt.hist(non_family_history_weights, edgecolor='black', linewidth=1.5, alpha = 0.3, label='Non Family History\n of Obesity Weight')
    plt.xlabel('Weight in kg')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    return family_history_obese_weights, non_family_history_weights
    
    
    
    
    