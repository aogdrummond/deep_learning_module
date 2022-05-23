import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import util.util as util
from util.evaluation import preprocess_df, show_soundfields, db, CSV_COLUMNS_NAMES

def evaluate_ssim_nmse(config_path:str):
    """
    Runs the evaluation of prediction performance through Structural
    Similarity (SSIM) and Normalized Mean Squared Error (NMSE) and saves
    inside ssion file a plot showing the results

    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    session_dir = config_path[:config_path.rfind('\\')+1]

    evaluation_path = os.path.join(session_dir, 'simulated_data_evaluation', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                  '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                  str(config['evaluation']['step_mics']))

    if not os.path.exists(evaluation_path): os.mkdir(evaluation_path)

    results_file_names = [file_name for file_name in os.listdir(evaluation_path) if file_name.endswith(".csv")]
    for results_file_name in results_file_names:
      results_path = os.path.join(evaluation_path,results_file_name)
      results_dataframe = pd.read_csv(results_path, names=CSV_COLUMNS_NAMES)
      dataframe_preprocessed = preprocess_df(results_dataframe)
      plot_evaluation(dataframe_preprocessed, 
                      sample_path = results_path)
    util.analyze_and_plot_simulated_results(evaluation_path,config,dB=True)
    
def compare_soundfields(config_path):
    """
    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('\\')+1]
    
    num_mics = config["visualization"]["num_mics"]
    visualization_path = os.path.join(session_dir,"visualization\\")
    
    show_soundfields(soundfield_path=visualization_path, 
                    freq_shown = config["evaluation"]["frequencies"])

def plot_evaluation(preprocessed_data:tuple, 
                    sample_path:str):

  print("Plotting individual performance")
  dict_loss_SSIM = preprocessed_data[0]
  dict_loss_NMSE = preprocessed_data[1]
  freq = preprocessed_data[2]
  all_labels = preprocessed_data[3]
  freq.sort(key=int)
  all_labels = np.sort(all_labels.astype(int))
  labels_idx = [0,5,9,12,15,18,21,26,31,39]
  labels_used = [all_labels[idx] for idx in labels_idx]

  plt.figure(figsize=(15,12))                                       
  
  keys = [mic for mic in dict_loss_SSIM.keys() if mic != 'num_mics']
  keys_leg = np.sort(np.array([int(mics) for mics in keys]))
  legend = [f"{n} mics" for n in keys_leg]
  session = sample_path.split("\\")[-4]
  sample_name = sample_path.split("\\")[-1]
  evaluation_path = sample_path[:sample_path.rfind("\\")+1]
  individual_results_folder = os.path.join(evaluation_path,"individual_performance")
  
  if not os.path.exists(individual_results_folder):
        os.mkdir(individual_results_folder)

  plt.subplot(211)

  for mic in keys:
    plotted_SSIM= []
    plotted_SSIM.extend(dict_loss_SSIM[mic][10:].tolist())
    plotted_SSIM.extend(dict_loss_SSIM[mic][:10].tolist())
    plt.plot(freq,plotted_SSIM,markevery=1,marker="o")


  plt.xticks(ticks=labels_idx,labels=labels_used,fontsize=20)
  plt.yticks(fontsize=20)
  plt.xlabel('Frequency [Hz]',fontsize=10)
  plt.ylabel('MSSIM',fontsize=30)
  plt.title(f'{session.capitalize()} | sample: {sample_name}',fontsize=20)
  plt.legend(legend,fontsize=15)
  plt.grid(color='k',linewidth=1)

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
  plt.title('',fontsize=15)
  plt.legend(legend,fontsize=15)
  plt.grid(color='k',linewidth=1)
  figure_name = "".join([sample_path.split("\\")[-1].split(".")[-2],".png"])
  plt.savefig(os.path.join(individual_results_folder,figure_name))
  plt.show()


