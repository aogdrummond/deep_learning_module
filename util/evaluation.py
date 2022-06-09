import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import os
import PIL
import json
import util.util as util

CSV_COLUMNS_NAMES=["name","num_file","xDim","yDim","m2","num_mics","num_comb","freq","NMSE","SSIM","pattern","p_real","p_predicted","p_previous"]

def preprocess_df(dataframe):
  
  num_mics = np.unique(dataframe['num_mics'].values)
  list_str_freq = np.unique(dataframe['freq'].values)
  freq = [freq for freq in list_str_freq if freq != 'freq']

  dict_loss_SSIM = {}
  dict_loss_NMSE = {}

  for mics in num_mics:
    data_SSIM = pd.DataFrame()
    data_NMSE = pd.DataFrame()

    for frequency in freq:
      df = dataframe[(dataframe['freq']==frequency) & (dataframe['num_mics']== mics)]
      values_NMSE = np.array(df['NMSE'],dtype=float)
      values_SSIM = np.array(df['SSIM'],dtype=float)
      data_SSIM[frequency] = pd.Series(values_SSIM.mean())
      data_NMSE[frequency] = pd.Series(values_NMSE.mean())

      dict_loss_SSIM[mics] = pd.DataFrame(data=np.array([values_SSIM.mean()]),columns=[frequency]).values  
      dict_loss_NMSE[mics] = pd.DataFrame(data=np.array([values_NMSE.mean()]),columns=[frequency]).values

    loss_SSIM = data_SSIM.values
    loss_NMSE = data_NMSE.values
    dict_loss_SSIM[mics] = np.reshape(loss_SSIM,(40,))
    dict_loss_NMSE[mics] = np.reshape(loss_NMSE,(40,))  

    labels = data_SSIM.columns

  return (dict_loss_SSIM,dict_loss_NMSE, freq, labels)


def normalize(sf_matrix:np.array,max_value:float,min_value:float)->np.array:
 
  sf_max = np.max(sf_matrix)
  sf_min = min_value #np.min(sf_matrix)
  normalized = np.divide(min_value + np.multiply((sf_matrix - sf_min),(max_value - min_value)),(sf_max - sf_min))

  return normalized

def compare_soundfields_to_num_mics(root_path:str,freq:list,n_mics:list=[]):
  """
  Plots the visualization of soundfield using different number of microphones,
  allowing it to be compared  
  Args:
  root_path: session's root path
  freq: Frequency to be plotted among those available
  n_mics: numbers of microphones whose visualization will be ploted. Default
  is all the mics in the folder
  """

  visualization_folders = [directory for directory in os.listdir(root_path) if directory.startswith("visualization_")]
  
  if len(n_mics) != 0:
    valid_folders = []
    for folder in visualization_folders:
      for n in n_mics:
        if n in folder:
          valid_folders.append(folder)
  
    visualization_folders = np.unique(valid_folders)
  
  ordened_mics_folder = []
  for vis_folder in visualization_folders:

    num_mics = int(vis_folder.split("_")[-1])
    ordened_mics_folder.append(num_mics)
  
  ordened_mics_folder = np.sort(ordened_mics_folder)
  ordened_vis_folder = ["".join(["visualization_",str(mics)]) for mics in ordened_mics_folder]
  for vis_folder in ordened_vis_folder:  
    mics = vis_folder.split("_")[-1]
    print(f"{mics} mics")
    show_soundfields("".join([root_path,"/",vis_folder,"/"]),freq)


def show_soundfields(soundfield_path:str,
                     freq_shown:list,
                     figsize:tuple = (6.4,4.8)):
  
  soundfield_files = os.listdir(soundfield_path)
  freq_list = sorted(freq_shown,key=int)
  GT = []
  Pred = []
  for f in freq_list:
    
    for filename in soundfield_files:
      if "Ground_Truth" in filename and filename.startswith(str(f)):
        GT.append(filename)
      if "Pred" in filename and filename.startswith(str(f)):
        Pred.append(filename)

  im_per_freq = []
  for i,j in zip(Pred,GT):
    im_per_freq.append(PIL.Image.open(soundfield_path+i))
    im_per_freq.append(PIL.Image.open(soundfield_path+j))
  
  valid_inputs = []
  for filename in GT:
    valid_inputs.append(filename.replace(GT[0][-20:],""))

  valid_inputs.sort(key=int)

  fig, ax = plt.subplots(nrows=2,ncols=len(GT),figsize=figsize)
  current_session = soundfield_path.split("/")[-3].split("_")[-1]
  fig.suptitle(f"Session {current_session}", fontsize=20)
  
  for idx in range(len(GT)):
    c=idx
    if c == 0:
      ax[0, c].imshow(np.asarray(im_per_freq[idx]))
      ax[0, c].set_xticks([])
      ax[0, c].set_yticks([])
      ax[0, c].set_title(f"{valid_inputs[idx]} Hz",fontweight="bold")
      ax[0, c].set_ylabel("PREDICTION",fontweight="bold")
      ax[1, c].imshow(np.asarray(im_per_freq[idx+1]))
      ax[1, c].set_xticks([])
      ax[1, c].set_yticks([])
      ax[1, c].set_ylabel("GROUND TRUTH",fontweight="bold")
    else:
      ax[0, c].imshow(np.asarray(im_per_freq[idx*2]))
      ax[0, c].set_xticks([])
      ax[0, c].set_yticks([])
      ax[0, c].set_title(f"{valid_inputs[idx]} Hz",fontweight="bold")
      ax[1, c].imshow(np.asarray(im_per_freq[(idx*2)+1]))
      ax[1, c].set_xticks([])
      ax[1, c].set_yticks([])
  
  insert_colorbar_in_image(fig = fig, 
                           average_spl = calculate_average_spl_per_freq(im_per_freq),
                           soundfield_path = soundfield_path)
  
  plt.tight_layout(rect=(0,0,0.86,1))
  plt.show()
  
def calculate_average_spl_per_freq(images:list)->list:
  """
  Calculates and average Sound Pressure Level(SPL) across the 
  frequencies of interest to allow plotting a colorbar representing 
  the scale of SPL for soundfield of all the frequencies plotted.
  Each soundfield corresponds to one element of list of images "images",
  which is averaged.
  """
  list_idx = [i*2 for i in range(int(len(images)/2))]
  spl_per_freq =  [np.sum(np.asarray(images[idx])[:,:,:3],axis=2) for idx in list_idx]
  spl_sum = 0
  for i in range(len(spl_per_freq)):
    spl_sum += spl_per_freq[i]
  
  average_spl = spl_sum/len(spl_per_freq)

  return average_spl

def insert_colorbar_in_image(fig: plt.figure, 
                            average_spl:list, 
                            soundfield_path:str):
  """
  Add a colorbar of magnitude according to scaled on "average_spl" list,
  to the plot being processed on "fig", ranging from the pressure as 
  registered on "pressure_range.json" file in "soundfield_path".
  """
  try:

    pressure_range_file = [filepath for filepath in os.listdir(soundfield_path) if filepath.endswith('.json')][0]
    pressure_range_path = "".join((soundfield_path,pressure_range_file))
    
    with open(pressure_range_path,"r") as json_file:
      pressure_range = json.load(json_file)

    max_pressure = pressure_range["MaxPressurePa"]
    min_pressure = pressure_range["MinPressurePa"]
        
    normalized_im = normalize(average_spl,max_pressure,min_pressure)
    db_imag = pa_to_db(normalized_im)
    max_SPL = int(np.floor(np.max(db_imag)))
    min_SPL = int(np.ceil(np.min(db_imag)))
    normalized_colormap = cm.ScalarMappable(colors.Normalize(max_SPL,min_SPL))
  
    cbar = fig.colorbar(mappable=normalized_colormap,
                        cax=plt.axes([0.85, 0.16, 0.045, 0.68]),
                        label="dB*", 
                        ticks = [min_SPL,83,85,87,89,91,max_SPL], 
                        orientation="vertical")

    cbar.ax.set_yticklabels([str(min_SPL),"83","85","87","89","91",str(max_SPL)])
    [tick_label.set_fontsize(14) for tick_label in cbar.ax.get_yticklabels()]

  except:

    print("Pressure Range was not found")  


def pa_to_db(sf_matrix:np.array,ref:float=20*10**(-6))->np.array:
 
  pressure_dB = 20*np.log10(abs(sf_matrix/ref))

  return pressure_dB

def plot_training_loss(history_path: str):
    """
    Plots loss and PSNR from the csv file
    from "path"
    """

    df = pd.read_csv(history_path)
    current_session = history_path.split("/")[-2]
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(df["epoch"], df["loss"], "b", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], "k", linewidth=2)
    plt.grid()
    plt.legend(["loss", "val_loss"], loc=5, prop={"size": 15})
    plt.title(f"Training Losses ({current_session})")
    plt.ylim([0, 5])
    plt.xlabel("Epoch")

    plt.subplot(2, 1, 2)
    plt.plot(df["epoch"], df["PSNR"], "r", linewidth=2)
    plt.plot(df["epoch"], df["val_PSNR"], "m", linewidth=2)
    plt.grid()
    plt.legend(["PSNR", "val_PSNR"], loc=5, prop={"size": 15})
    plt.title(f"Training PSNR ({current_session})")
    plt.ylim([10, 20])
    plt.xlabel("Epoch")

    config_path = os.path.join(history_path[: history_path.rfind("/")], "config.json")
    config = util.load_config(config_path)
    save_path = os.path.join(
        history_path[: history_path.rfind("/")],
        "simulated_data_evaluation",
        "min_mics_"
        + str(config["evaluation"]["min_mics"])
        + "_max_mics_"
        + str(config["evaluation"]["max_mics"])
        + "_step_mics_"
        + str(config["evaluation"]["step_mics"]),
    )

    plt.savefig(os.path.join(save_path, "session_loss_history.png"))


