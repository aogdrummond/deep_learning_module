import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import util.util as util
from util.eval_util import plot_training_loss, preprocess_df
from util.util_analysis import analyze_and_plot_simulated_results

def db(dict_loss):

  loss = np.array(dict_loss)
  ref = loss[~np.isnan(loss)]
  ref = np.max(ref)/0.6
  dict_in_dB = 20*np.log10(abs(np.array(dict_loss)/ref))

  return dict_in_dB

CSV_COLUMNS_NAMES=["name","num_file","xDim","yDim","m2","num_mics","num_comb","freq","NMSE","SSIM","pattern","p_real","p_predicted","p_previous"]

def evaluate_ssim_nmse(config_path:str):
    """
    Runs the evaluation of prediction performance through Structural
    Similarity (SSIM) and Normalized Mean Squared Error (NMSE) and saves
    inside ssion file a plot showing the results

    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('\\')+1]
    evaluation_path = "".join([session_dir, 'simulated_data_evaluation\\', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                  '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                  str(config['evaluation']['step_mics'])])
    results_file_name = random.choice(os.listdir(evaluation_path))
    results_path = "".join([evaluation_path,"\\",results_file_name])
    results_dataframe = pd.read_csv(results_path, names=CSV_COLUMNS_NAMES)
    dataframe_preprocessed = preprocess_df(results_dataframe)
    session_history_path = "".join([session_dir,"history_session_",str(config["training"]["session_id"]),".csv"])
    plot_training_loss(history_path=session_history_path)
    plot_evaluation(dataframe_preprocessed, 
                    path = results_path)

def plot_average_results(config_path):

    """
    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('\\')+1]
    evaluation_path = "".join([session_dir, 'simulated_data_evaluation\\', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                  '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                  str(config['evaluation']['step_mics'])])
    
    analyze_and_plot_simulated_results(evaluation_path,config,dB=True)

    

def plot_evaluation(preprocessed_data, 
                    num_mics=None, 
                    path=None):


  dict_loss_SSIM = preprocessed_data[0]
  dict_loss_NMSE = preprocessed_data[1]
  freq = preprocessed_data[2]
  labels = preprocessed_data[3]
  freq = [int(f) for f in freq]
  freq.sort()
  freq = [str(f) for f in freq]
  empty_array = np.empty(len(labels))    #Placeholder
  for i, label in enumerate(labels):
    label = int(label)
    empty_array[i] = label

  labels = np.sort(empty_array.astype(int))
  labels_idx = [0,5,9,12,15,17,19,20,31,39]
  labels_used = [labels[idx] for idx in labels_idx]
  #SSIM
  plt.figure(figsize=(15,12))                                       
  plt.subplot(211)

  keys = [mic for mic in dict_loss_SSIM.keys() if mic != 'num_mics']
  keys_leg = np.sort(np.array([int(elemento) for elemento in keys]))
  for mic in keys:
    plotted_SSIM= []
    plotted_SSIM.extend(dict_loss_SSIM[mic][10:].tolist())
    plotted_SSIM.extend(dict_loss_SSIM[mic][:10].tolist())
    plt.plot(freq,plotted_SSIM,markevery=1,marker="o")

  legend = []
  keys_leg = np.sort(np.array([int(elemento) for elemento in keys]))
  for n in keys_leg:
    legend.append(f'{n} mics')
  plt.xticks(ticks=labels_idx,labels=labels_used,fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel('Frequency [Hz]',fontsize=10)
  plt.ylabel('MSSIM',fontsize=30)
  if path != None:
    session = path.split("\\")[-4]
    sample = path.split("\\")[-1]
    plt.title(f'{session.capitalize()} | sample: {sample}',fontsize=20)
  plt.legend(legend,fontsize=15)
  plt.grid(color='k',linewidth=1)

  #NMSE
  plt.subplot(212)

  for mic in keys:
    plotted_NMSE= []
    plotted_NMSE.extend(dict_loss_NMSE[mic][10:].tolist())
    plotted_NMSE.extend(dict_loss_NMSE[mic][:10].tolist())
    plt.plot(freq,db(plotted_NMSE),markevery=1,marker="o")

  plt.xticks(ticks=labels_idx,labels=labels_used,fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel('Frequency [Hz]',fontsize=10)
  plt.ylabel('NMSE(dB)',fontsize=25)
  plt.legend(legend,fontsize=15)
  plt.grid(color='k',linewidth=1)
  plt.title('',fontsize=15)
  plt.show()


