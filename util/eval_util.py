import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
import scipy.io
import math
import os

columns_names=["name","num_file","xDim","yDim","m2","num_mics","num_comb","freq","NMSE","SSIM","pattern","p_real","p_predicted","p_previous"]

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


def plot_evaluation(dicts, freq, labels,num_mics=None,path=None):


  dict_loss_SSIM = dicts[0]
  dict_loss_NMSE = dicts[1]
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

  if type(num_mics) == list:
    keys = [mic for mic in num_mics]
    keys_leg = np.sort(np.array([int(elemento) for elemento in keys]))
  else:
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
    session = path.split("/")[-4].split("_")[-1]
    sample = path.split("/")[-1].split(".")[0].split("_")[-1]
  plt.title(f'Session {session} - sample {sample} - NMSE',fontsize=20)
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


def db(dict_loss):

  loss = np.array(dict_loss)
  ref = loss[~np.isnan(loss)]
  ref = np.max(ref)/0.6
  # ref = 0.6
  # import pdb
  # pdb.set_trace()
  dict_in_dB = 20*np.log10(abs(np.array(dict_loss)/ref))

  return dict_in_dB

def room_properties(path,source):
    """
    Calculates expected properties for the room baseado on its dimensions
    alfa is calculated according to Eyring's formula
    Args:
        path(str):path to the file for the properties to be extracted
    Returns:
        S_planta(flt): Transversal area (floors)
        S(flt): Whole room's area
        V(flt): Room's volume
        alfa(flt): Room's mean absorption
    """
    mat = scipy.io.loadmat(path)
    dim1 = mat["Setup"]["Room"][0][0][0][0][0][0][0]
    dim2 = mat["Setup"]["Room"][0][0][0][0][0][0][1]
    height = mat["Setup"]["Room"][0][0][0][0][0][0][2]
    S_planta = dim1 * dim2
    S = 2 * S_planta + 2*dim1*height + 2*dim2*height
    S = round(S,2)
    V = S_planta * height
    V = round(V,2)
    if source == 'femder':
      T60 = mat["Setup"]["Room"][0][0][0][0][2][0][0]    
    elif source == "matlab":
      T60 = mat["Setup"]["Room"][0][0][0][0][1][0][0] 
    else:
      raise ValueError("Invalid source")
    alfa = 1 - math.exp(0.161*V/(-S*T60))
    alfa = round(alfa,3)
    
    return S_planta, S, V, alfa


def plot_training_loss(history_path):
    """
    Plots loss and PSNR from the csv file
    from "path"
    """
    df = pd.read_csv(history_path)
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    x = df["epoch"]
    y = df["loss"]
    plt.plot(x,y,"b",linewidth=2)
    y = df["val_loss"]
    plt.plot(x,y,"k",linewidth=2)
    plt.grid()
    plt.legend(["loss","val_loss"],loc=5, prop={'size': 15})
    current_session = history_path.split("\\")[-2]
    plt.title(f"Training Losses ({current_session})")
    plt.ylim([0,5])
    plt.xlabel("Epoch")
    plt.subplot(2,1,2)
    y = df["PSNR"]
    plt.plot(x,y,"r",linewidth=2)
    y = df["val_PSNR"]
    plt.plot(x,y,"m",linewidth=2)
    plt.ylim([10,20])
    plt.grid()
    plt.legend(["PSNR","val_PSNR"],loc=5, prop={'size': 15})
    plt.title(f"Training PSNR ({current_session})")
    plt.xlabel("Epoch")

def show_soundfields(path,
                     freq,
                     figsize = (6.4,4.8)):

  import PIL
  import os
  import json

  filename = os.listdir(path)
   
  freq_list = freq.split(",")
  freq_list = sorted(freq_list,key=int)
  GT = []
  Pred = []
  for f in freq_list:
    
    for element in filename:
      if "Ground_Truth" in element and element.startswith(f):
        GT.append(element)
      if "Pred" in element and element.startswith(f):
        Pred.append(element)

  im = []
  for i,j in zip(Pred,GT):
    im.append(PIL.Image.open(path+i))
    im.append(PIL.Image.open(path+j))

  print(Pred)
  print(GT)


  valid_inputs = []
  for element in GT:
    valid_inputs.append(element.replace(GT[0][-20:],""))

  valid_inputs = sorted(valid_inputs,key=int)
  fig, ax = plt.subplots(nrows=2,ncols=len(GT),figsize=figsize)
  
  cmap = colors.Colormap("jet")

  for idx in range(len(GT)):
    
    c=idx
    if c == 0:
      ax[0, c].imshow(np.asarray(im[idx]))
      ax[0, c].set_xticks([])
      ax[0, c].set_yticks([])
      ax[0, c].set_title(f"{valid_inputs[idx]} Hz",fontweight="bold")
      ax[0, c].set_ylabel("PREDICTION",fontweight="bold")
      ax[1, c].imshow(np.asarray(im[idx+1]))
      ax[1, c].set_xticks([])
      ax[1, c].set_yticks([])
      ax[1, c].set_ylabel("GROUND TRUTH",fontweight="bold")
    else:
      ax[0, c].imshow(np.asarray(im[idx*2]))
      ax[0, c].set_xticks([])
      ax[0, c].set_yticks([])
      ax[0, c].set_title(f"{valid_inputs[idx]} Hz",fontweight="bold")
      ax[1, c].imshow(np.asarray(im[(idx*2)+1]))
      ax[1, c].set_xticks([])
      ax[1, c].set_yticks([])
  
  #Ler valores do json
  try:
    json_name = [filepath for filepath in filename if filepath.endswith('.json')][0]
    json_path = "".join((path,json_name))
  
    accumulator = []
    for idx in [0,2,4,6]:
      sum_rgb = np.sum(np.asarray(im[idx])[:,:,:3],axis=2)
      accumulator.append(sum_rgb)
    
    sums_mean = (accumulator[0]+accumulator[1]+accumulator[2]+accumulator[3])/4

    with open(json_path,"r") as json_file:
      pressure_range = json.load(json_file)
    max_pressure = pressure_range["MaxPressurePa"]
    min_pressure = pressure_range["MinPressurePa"]
    
    # #Transformar em db
    # #Inserir label no colorbar
    normalized_im = normalize(sums_mean,max_pressure,min_pressure)
    db_imag = pa_to_db(normalized_im)
    # imag = plt.imshow(db_imag)
    max_SPL = int(np.floor(np.max(db_imag)))
    min_SPL = int(np.ceil(np.min(db_imag)))
    norm = colors.Normalize(max_SPL,min_SPL)
    imag = cm.ScalarMappable(norm)

    cbar = fig.colorbar(imag,cax=plt.axes([0.95, 0.16, 0.045, 0.68]),label="dB*",ticks = [min_SPL,83,85,87,89,91,max_SPL],orientation="vertical")
    cbar.ax.set_yticklabels([str(min_SPL),"83","85","87","89","91",str(max_SPL)])
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(14)
  except:
    print("Pressure Range was not found")
  print(path.split("/")[-2])
  plt.show()

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


def normalize(sf_matrix:np.array,max_value:float,min_value:float)->np.array:
 
  sf_max = np.max(sf_matrix)
  sf_min = min_value #np.min(sf_matrix)
  normalized = np.divide(min_value + np.multiply((sf_matrix - sf_min),(max_value - min_value)),(sf_max - sf_min))

  return normalized


def pa_to_db(sf_matrix:np.array,ref:float=20*10**(-6))->np.array:
 
  pressure_dB = 20*np.log10(abs(sf_matrix/ref))

  return pressure_dB

