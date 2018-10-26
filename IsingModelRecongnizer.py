#!/usr/bin/python 
#
# Author: Liyang
# Date: 2018.10.24
# Descripution: 
#   This program is designed for using the Deep Learning method to 
#     reconginze the phase transition point in 2D Ising model. 
# 

###################
### Import Part ###
###################

import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import sys, os

################################
### Set the Basic Parameters ###
################################
kSpinDown = -1
kSpinUp = 1
kDefaultPatternSize = 50
kDefaultTrainDataQuantity = 10000
kDefaultTestDataQuantity = 1000
kEpochs = 20
kTransitionTemperature = 2.3
kTemperatureUpperLimit = 8.0
kTemperatureLowerLimit = 1.0
kIsTransition = [1,0]
kNotTransition = [0,1]
kIsingHeatSteps = 500000
kModelSaveFileName = '_ising_model_save.h5'
kTrainDataSaveFileName = '_ising_train_data.txt'
kTestDataSaveFileName = '_ising_test_data.txt'
kTrainLabelsSaveFileName = '_ising_train_labels.txt'
kTestLabelsSaveFileName = '_ising_test_labels.txt'

###########################
### Define the Function ###
###########################
def welcome_interface():
  return 0

def parameters_reader():
  ## Whether load data file
  if os.path.isfile(kTestLabelsSaveFileName):
    print('Load train/test data from file?(Y/N)')
    load_data_from_file = input('> ')
    if 'Y' == load_data_from_file or 'y' == load_data_from_file:
      load_data_from_file = True
    else:
      load_data_from_file = False
  else:
    load_data_from_file = False
  ## pattern size
  if load_data_from_file:
    temp_data = np.loadtxt(kTestDataSaveFileName, dtype=int)
    pattern_size = len(temp_data[0])
    print('Using the data of Pattern Size:',pattern_size)
  else:
    print('Input the pattern size (Default',kDefaultPatternSize,').')
    pattern_size = input('> ')
    if pattern_size == '':
      pattern_size = kDefaultPatternSize
    else:
      pattern_size = int(pattern_size)
  ## Train data quantity
  train_data_quantity = kDefaultTrainDataQuantity
  ## Test data quantity
  test_data_quantity = kDefaultTestDataQuantity

  return pattern_size, \
         train_data_quantity, \
         test_data_quantity, \
         load_data_from_file

def generate_ising_pattern(pattern_size, temperature):
  ising_pattern = np.ones((pattern_size+2,pattern_size+2),dtype=int)
  for loop_index in range(kIsingHeatSteps):
    ## Periodic bound condition
    ising_pattern[0,:] = ising_pattern[pattern_size,:]
    ising_pattern[pattern_size+1,:] = ising_pattern[1,:]
    ising_pattern[:,0] = ising_pattern[:,pattern_size]
    ising_pattern[:,pattern_size+1] = ising_pattern[:,1]

    pickup_row_index = random.randint(1,pattern_size)
    pickup_column_index = random.randint(1,pattern_size)
    
    Energy_berfore_filp = \
      -ising_pattern[pickup_row_index,pickup_column_index] * \
      (ising_pattern[pickup_row_index+1,pickup_column_index] + \
       ising_pattern[pickup_row_index,pickup_column_index+1] + \
       ising_pattern[pickup_row_index-1,pickup_column_index] + \
       ising_pattern[pickup_row_index,pickup_column_index-1])
    if Energy_berfore_filp < 0:
      accpect_filp_possibility = \
        math.exp(2 * Energy_berfore_filp / temperature)
      if random.random() > accpect_filp_possibility:
        continue
    ising_pattern[pickup_row_index,pickup_column_index] = \
      -ising_pattern[pickup_row_index,pickup_column_index]
  ising_pattern = np.delete(ising_pattern,pattern_size+1,0)
  ising_pattern = np.delete(ising_pattern,pattern_size+1,1)
  ising_pattern = np.delete(ising_pattern,0,0)
  ising_pattern = np.delete(ising_pattern,0,1)
  return ising_pattern

def get_ising_pattern_data(data_quantity, pattern_size):
  is_transition_set = []
  ising_pattern_set = []
  for loop_index in range(data_quantity):
    ## Print the Prosscee information
    pross_information = \
      'Data Generating... ('+str(loop_index+1)+'/'+str(data_quantity)+')'
    pross_information_length = len(pross_information)
    sys.stdout.write(pross_information)
    sys.stdout.flush()
    sys.stdout.write('\b'*pross_information_length)
    ## Generate the ising pattern 
    is_transition = random.choice([True, False])
    if is_transition:
      temperature = random.uniform(kTransitionTemperature,
                                   kTemperatureUpperLimit)
      is_transition_set.append(kIsTransition)
    else:
      temperature = random.uniform(kTransitionTemperature,
                                   kTemperatureUpperLimit)
      is_transition_set.append(kNotTransition)

    ising_pattern_set.append(generate_ising_pattern(pattern_size,
                                                    temperature))
  ising_pattern_set = np.array(ising_pattern_set)
  is_transition_set = np.array(is_transition_set)
  print('')
  print('')
  return ising_pattern_set, is_transition_set

def deep_learning_process(input_layer_size, 
                          train_data, 
                          train_labels,
                          test_data,
                          test_labels):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(input_layer_size,
                                  activation=tf.nn.sigmoid))
  model.add(tf.keras.layers.Dense(input_layer_size//2,
                                  activation=tf.nn.sigmoid))
  model.add(tf.keras.layers.Dense(input_layer_size//4,
                                  activation=tf.nn.sigmoid))
  model.add(tf.keras.layers.Dense(input_layer_size//8,
                                  activation=tf.nn.sigmoid))
  model.add(tf.keras.layers.Dense(input_layer_size//16,
                                  activation=tf.nn.sigmoid))
  model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(train_data, train_labels, epochs=kEpochs)
  model.save(kModelSaveFileName)
  print('')
  print('Trained Model save in File:',kModelSaveFileName)
  val_loss, val_acc = model.evaluate(test_data, test_labels)
  print('')
  print("Validation Loss      : ",val_loss) 
  print("Validation Accuracy  : ",val_acc)

def data_save_to_file(data, file_name):
  if len(data.shape) < 3:
    np.savetxt(file_name, data, fmt='%d')
  else:
    with open(file_name, 'w') as outfile:
      outfile.write('# Array shape: {0}\n'.format(data.shape))
      for data_slice in data:
        np.savetxt(outfile, data_slice, fmt='%d')
        outfile.write('# New slice\n')
  return 0

def data_load_from_file(file_name, origin_data_quantity):
  data = np.loadtxt(file_name, dtype=int)
  if len(data) != origin_data_quantity:
    single_data_size = len(data) // origin_data_quantity
    data = data.reshape(origin_data_quantity, 
                        single_data_size, single_data_size)
  return data

def main():
  welcome_interface()
  pattern_size, \
  train_data_quantity, \
  test_data_quantity ,\
  load_data_from_file = parameters_reader()
  if load_data_from_file:
    print('>>> Loading Train Data... <<<')
    train_data = data_load_from_file(kTrainDataSaveFileName,
                                     train_data_quantity)
    train_labels = data_load_from_file(kTrainLabelsSaveFileName,
                                       train_data_quantity)
    print('>>> Loading Test Data... <<<')
    test_data = data_load_from_file(kTestDataSaveFileName,
                                    test_data_quantity)
    test_labels = data_load_from_file(kTestLabelsSaveFileName,
                                      test_data_quantity)
  else:
    print('>>> Generate Train Data... <<<')
    train_data, train_labels \
      = get_ising_pattern_data(train_data_quantity, pattern_size)
    data_save_to_file(train_data, kTrainDataSaveFileName)
    data_save_to_file(train_labels, kTrainLabelsSaveFileName)
    data_save_to_file(train_data, 
                      '_'+str(pattern_size)+kTrainDataSaveFileName)
    data_save_to_file(train_labels, 
                      '_'+str(pattern_size)+kTrainLabelsSaveFileName)
    print('>>> Generate Test Data... <<<')
    test_data, test_labels \
      = get_ising_pattern_data(test_data_quantity, pattern_size)
    data_save_to_file(test_data, kTestDataSaveFileName)
    data_save_to_file(test_labels, kTestLabelsSaveFileName)
    data_save_to_file(test_data, 
                      '_'+str(pattern_size)+kTestDataSaveFileName)
    data_save_to_file(test_labels, 
                      '_'+str(pattern_size)+kTestLabelsSaveFileName)
  print('>>> Deep Learning Process... <<<')
  input_layer_size = pattern_size ** 2
  deep_learning_process(input_layer_size, 
                        train_data, 
                        train_labels,
                        test_data,
                        test_labels)

  return 0

####################
### Main Process ###
####################

if __name__ == '__main__':
  main()