import argparse as ap
import json
import os
import time

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.utils import try_import_tf
from ray.tune.logger import pretty_print

from Experiments import TaxiRebLite, TaxiRebalance, get_CLI_options, \
    RESULTS, CONFIG, update_graph_file, update_vehicle_initial_distribution

tf = try_import_tf()
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')


if __name__ == '__main__':

    # unique results directory for every run
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    RESULTS = os.path.join(RESULTS, curr_time)
    # Config file has priority over CLI arguments
    args = get_CLI_options()

    try:
        with open(args.config, 'r') as file:
            file_conf = json.load(file)
    except FileNotFoundError:
        file_conf = None
        if args.config != 'None':
            raise

    # NODES = sorted(pd.read_csv(os.path.join(CONFIG, 'aam.csv'), index_col=0, header=0).index.values.tolist())
    NODES = sorted([236, 237, 186, 170, 141, 162, 140, 238, 142, 229, 239, 48, 161, 107, 263, 262, 234, 68, 100, 143])
    # NODES = sorted([236, 237, 186, 170, 141])
    initial_vehicle = args.init_veh
    iterations = args.iter
    vehicle_speed = args.veh_speed
    use_lite = args.lite
    if file_conf is not None:
        NODES = sorted(file_conf.pop("nodes", NODES))
        initial_vehicle = int(file_conf.pop("init_veh", initial_vehicle))
        iterations = int(file_conf.pop("iter", iterations))
        vehicle_speed = int(file_conf.pop("veh_speed", vehicle_speed))
        use_lite = bool(file_conf.pop('use_lite', use_lite))

    update_graph_file(NODES, os.path.join(CONFIG, 'gps.csv'), os.path.join(CONFIG, 'aam.csv'))
    update_vehicle_initial_distribution(nodes=NODES, veh_dist=[initial_vehicle for i in range(len(NODES))])

    ray.init()
    nodes_list = [str(x) for x in NODES]
    configure = ppo.DEFAULT_CONFIG.copy()
    if not use_lite:
        configure['env'] = TaxiRebalance
    else:
        print('using lite')
        configure['env'] = TaxiRebLite

    configure['model']['fcnet_hiddens'] = [256, 256]
    configure['num_workers'] = args.num_cpu if args.num_cpu is not None else 1
    configure['num_gpus'] = args.num_gpu if args.num_gpu is not None else 0
    configure['vf_clip_param'] = args.vf_clip
    configure['lr'] = args.lr
    configure['train_batch_size'] = args.tr_bat_size
    configure['rollout_fragment_length'] = args.wkr_smpl_size
    configure['sgd_minibatch_size'] = args.sgd_bat_size
    configure['env_config'] = {
        "start_time": '08:00:00',
        "time_horizon": 10,  # hours
        "max_vehicle": 500000,
        "reb_interval": 600,  # seconds 60 steps per episode
        "max_travel_time": 1000,
        "max_passenger": 1e6,
        "nodes_list": nodes_list,
        "near_neighbor": args.num_neighbor,
        "plot_queue_len": False,  # do not use plot function for now
        "dispatch_rate": args.dpr,
        "alpha": args.alpha,
        "beta": args.beta,
        "sigma": args.sigma,
        "save_res_every_ep": 100,
        "veh_speed": vehicle_speed
    }

    if file_conf is not None:
        env_config = file_conf.pop('env_config', None)
        model_config = file_conf.pop('model', None)
        configure.update(file_conf)
        if env_config is not None:
            configure['env_config'].update(env_config)
        if model_config is not None:
            configure['model'].update(model_config)

    stt = time.time()
    trainer = ppo.PPOTrainer(config=configure)

    for _ in range(iterations):
        print('Iteration:', _+1)
        results = trainer.train()
        if (_+1) % 100 == 0:
            print(pretty_print(results))

    check_pt = trainer.save()
    print(f"Model saved at {check_pt}")
    print(time.time()-stt)

    policy = trainer.get_policy()


