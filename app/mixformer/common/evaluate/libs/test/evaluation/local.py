from libs.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    
    # Set your local paths here.
    settings.got10k_path = '../../../../data/evaluate_dataset/GOT_10K'
    settings.result_plot_path = '../../mixformer/tmp_res/result_plots'
    settings.results_path = '../../mixformer/tmp_res/tracking_results'

    return settings

