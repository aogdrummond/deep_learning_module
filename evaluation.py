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
                    sample_path = results_path)
    util.analyze_and_plot_simulated_results(evaluation_path,config,dB=True)
    
def compare_soundfields(config_path):
    """
    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind('\\')+1]

    num_mics = config["visualization"]["num_mics"]
    visualization_path = "".join([session_dir,f"visualization_{num_mics}_mics\\"])
    
    show_soundfields(soundfield_path=visualization_path, 
                    freq_shown = config["evaluation"]["frequencies"])


def plot_training_loss(history_path:str):
    """
    Plots loss and PSNR from the csv file
    from "path"
    """
    df = pd.read_csv(history_path)
    current_session = history_path.split("\\")[-2]
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(df["epoch"],df["loss"],"b",linewidth=2)
    plt.plot(df["epoch"],df["val_loss"],"k",linewidth=2)
    plt.grid()
    plt.legend(["loss","val_loss"],loc=5, prop={'size': 15})
    plt.title(f"Training Losses ({current_session})")
    plt.ylim([0,5])
    plt.xlabel("Epoch")
    
    plt.subplot(2,1,2)
    plt.plot(df["epoch"],df["PSNR"],"r",linewidth=2)
    plt.plot(df["epoch"],df["val_PSNR"],"m",linewidth=2)
    plt.grid()
    plt.legend(["PSNR","val_PSNR"],loc=5, prop={'size': 15})
    plt.title(f"Training PSNR ({current_session})")
    plt.ylim([10,20])
    plt.xlabel("Epoch")
    individual_results_path = "".join([history_path[:history_path.rfind("\\")],"\\", 
    "simulated_data_evaluation\\min_mics_5_max_mics_65_step_mics_15\\individual_performance\\"])
    if not os.path.exists(individual_results_path):
        os.mkdir(individual_results_path)
    plt.savefig("".join([individual_results_path,"\\","session_loss_history.png"]))


def plot_evaluation(preprocessed_data:tuple, 
                    sample_path:str):


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
  individual_results_path = "".join([evaluation_path,"\\","individual_performance"])
  if not os.path.exists(individual_results_path):
        os.mkdir(individual_results_path)

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
  
  plt.savefig("".join([individual_results_path,"\\",sample_path.split("\\")[-1].split(".")[-2],".png"]))
  plt.show()


