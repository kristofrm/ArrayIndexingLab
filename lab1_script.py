# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:04:55 2024

Analysis of the Estimation of Obesity Levels Based On Eating Habits and Physical Condition

This script uses the files from https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
to see if there is a connection between people who have a family history of high weight/obesity and having a higher weight than people who do not 
have this history.

This dataset includes data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, 
based on their eating habits and physical condition. The data contains 17 attributes and 2111 records, the records are labeled 
with the class variable NObesity (Obesity Level), that allows classification of the data using the values of Insufficient Weight, 
Normal Weight, Overweight Level I, Overweight Level II, Obesity Type I, Obesity Type II and Obesity Type III. 77% of the data 
was generated synthetically using the Weka tool and the SMOTE filter, 23% of the data was collected directly from users through 
a web platform.

Contributors/sources: 
    https://www.practiceprobs.com/problemsets/python-numpy/intermediate/#

@author: Lauren Downs and Kristof Rohaly-Medved
"""
#%% Imported Modules
import numpy as np
from matplotlib import pyplot as plt

#%% Part 1: Finding the file and printing Details

# File path and name variables
data_file = 'ObesityDataSet.txt' 

# Description
print('Description:\n------------')
print('The file {data_file} contains data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia, based on their eating habits and physical condition. ')
print('\nHypothesis:\n------------')
print("Using this data, we plan to evaluate if the patient has family members who are and have been overweight (family history), then the patient's weight is more likely to be higher than those who have no family history.")

#%% Part 2: Loading the data
obesity_data_array = np.loadtxt(data_file, dtype = str)
print(f'The number of examples in this data set is {obesity_data_array.shape[0]-1}. \nThe number of features in this data set is {obesity_data_array.shape[1]}.') # 1 is subtracted because the headers of the column are the first index

#%% Part 3: Extracting Variables of Interest

# indexing data array - the first row is skipped because it contains headers; headers are saved in separate variables

patient_weigth_array_str = obesity_data_array[1:,3]
patient_weigth_header = obesity_data_array[0,3]
patient_fam_history_array = obesity_data_array[1:,4]
patient_fam_history_header = obesity_data_array[0,4]

# converting numerical array to float
patient_weigth_array = np.array(patient_weigth_array_str, dtype = float)

# printing out the data
for example_index, example in enumerate(patient_weigth_array):
    print(f'example {example_index}: {patient_weigth_header} = {patient_weigth_array[example_index]} kg, {patient_fam_history_header} = {patient_fam_history_array[example_index]} ')
    
# Boolean indexing and print out
# help from https://www.practiceprobs.com/problemsets/python-numpy/intermediate/#
is_fam_history_array = patient_fam_history_array == 'yes'
for example_index, example in enumerate(patient_weigth_array[is_fam_history_array]):
    print(f'Has a family history of obesity {example_index}: {patient_weigth_header} = {patient_weigth_array[is_fam_history_array][example_index]} kg')
    
# variables based on Boolean indexing
fam_history_weight_array = patient_weigth_array[is_fam_history_array]
no_fam_history_weight_array = patient_weigth_array[~is_fam_history_array]

#%% Part 4: Plot variables

# Create figure and add title
plt.figure(1, clear=True)
plt.suptitle('Correlation Between Family History \nof Obesity and Weight', fontsize=16, fontweight='bold')

# Plot histograms for male and female weights
plt.hist(fam_history_weight_array, edgecolor='black', linewidth=1.5, alpha = 0.3, label='Family History Obese')
plt.hist(no_fam_history_weight_array, edgecolor='black', linewidth=1.5, alpha = 0.3, label='Non Family History Obese')
plt.xlabel('Weight in kg')
plt.ylabel('Number of Individuals')
plt.legend()
plt.grid()
plt.tight_layout()

# Printing relation to hypothesis
print('This plot supports the hypothesis because the histogram clearly shows that there are more people with a higher weight who have a family history of high weight/obesity. In contrast, people who had no family history trended towards having lower weights.')

#%% Part 5: Declare and call a function to replicate your findings

import lab1_module as l1m

# Split weights and family history into 2 separate data groups
even_weights = patient_weigth_array[::2]
even_family_history = patient_fam_history_array[::2]
odd_weights = patient_weigth_array[1::2]
odd_family_history = patient_fam_history_array[1::2]


# Set up new figure with title
plt.figure(2, clear=True)
plt.suptitle('Correlation Between Family History of \nObesity and Weight for 2 Groups', fontsize=16, fontweight='bold')

# Plot histograms for the first subset
plt.subplot(1, 2, 1)
plt.title('Group 1 Results')
even_family_obese_weights, even_non_family_obese_weights = l1m.plot_histogram(even_weights, even_family_history)


# Plot histograms for the second subset
plt.subplot(1, 2, 2)
plt.title('Group 2 Results')
odd_family_obese_weights, odd_non_family_obese_weights = l1m.plot_histogram(odd_weights, odd_family_history)

#%% Part 6: Saving the Results

# Saving features and labels
np.savetxt('data_features.txt',patient_weigth_array, fmt= '%.7f')
np.savetxt('data_labels.txt',patient_fam_history_array, fmt = '%s')

# Re-loading features and labels
patient_weigth_array_loaded = np.loadtxt('data_features.txt', dtype = float)
patient_fam_history_array_loaded = np.loadtxt('data_labels.txt', dtype = str)

# Comparing the files to the data
# features
if np.array_equal(patient_weigth_array_loaded, patient_weigth_array):
    print('The features file array matches the features data array')
else:
    print('The feature arrays do not match')
    if patient_weigth_array_loaded.shape != patient_weigth_array.shape: # prints one error for shape differences 
        print(f'The loaded array has shape {patient_weigth_array_loaded.shape} and the original array has shape {patient_weigth_array.shape}')
    else: # prints one error for value differences
        print(f'The values that do not match are {patient_weigth_array_loaded [patient_weigth_array_loaded != patient_weigth_array]} in the saved file. (Original values are {patient_weigth_array [patient_weigth_array_loaded != patient_weigth_array]})')
# labels
if np.array_equal(patient_fam_history_array_loaded, patient_fam_history_array):
    print('The labels file array matches the labels data array')
else:
    print('The label arrays do not match')
    if patient_fam_history_array_loaded.shape != patient_fam_history_array.shape: # prints one error for shape differences
        print(f'The loaded array has shape {patient_fam_history_array_loaded.shape} and the original array has shape {patient_fam_history_array.shape}')
    else: # prints one error for value differences
        print(f'The values that do not match are {patient_fam_history_array_loaded [patient_fam_history_array_loaded != patient_fam_history_array]} in the saved file. (Original values are {patient_fam_history_array [patient_fam_history_array_loaded != patient_fam_history_array]})')

# Saving the figures
plt.figure(1)
plt.savefig('figure_all.pdf')
plt.figure(2)
plt.savefig('figure_halved.pdf')

