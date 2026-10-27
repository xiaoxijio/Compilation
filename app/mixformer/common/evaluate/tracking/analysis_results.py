import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from libs.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from libs.test.evaluation import get_dataset
from libs.test.evaluation.environment import env_settings
import argparse

class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False, tracker_params=None):
        # assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name
        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


trackers = []
parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
parser.add_argument('--tracker_name', type=str, default=None, help='Name of config file.')
parser.add_argument('--tracker_param', type=str, help='Name of config file.')
parser.add_argument('--dataset_name', type=str, help='Name of config file.')
parser.add_argument('--run_ids', nargs='+', help='<Required> Set flag', required=False)
args = parser.parse_args()

dataset_name = args.dataset_name

if args.tracker_name is None:
    args.tracker_name = 'mixformer2_vit_online'

trackers.extend(trackerlist(name=args.tracker_name, parameter_name=args.tracker_param, dataset_name=args.dataset_name,
                            run_ids=args.run_ids, display_name='MixFormerV2'))

dataset = get_dataset(dataset_name)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
