import os
import util.util as util
from util.evaluation import show_soundfields

def evaluate_general_ssim_nmse(config_path:str):
    """
    Runs the evaluation of prediction performance through Structural
    Similarity (SSIM) and Normalized Mean Squared Error (NMSE) and saves
    inside ssion file a plot showing the results

    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)
    session_dir = config_path[:config_path.rfind('/')+1]
    evaluation_path = os.path.join(session_dir, 'simulated_data_evaluation', 'min_mics_' + str(config['evaluation']['min_mics']) +
                                  '_max_mics_' + str(config['evaluation']['max_mics']) + '_step_mics_' +
                                  str(config['evaluation']['step_mics'])).replace("\\","/")

    if not os.path.exists(evaluation_path): os.mkdir(evaluation_path)

    util.analyze_and_plot_simulated_results(evaluation_path,config,dB=True)
    
def compare_soundfields(config_path):
    """
    Plot side-by-side the ground-truth and predicted soundfields,
    for the frequencies set on "config.json"
    """
    config = util.load_config(config_path)
    print('Loaded configuration from: %s' % config_path)

    session_dir = config_path[:config_path.rfind("/")+1]
    visualization_path = os.path.join(session_dir,"visualization/")
    show_soundfields(soundfield_path=visualization_path, 
                    freq_shown = config["evaluation"]["frequencies"])

