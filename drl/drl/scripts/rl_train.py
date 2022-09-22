import argparse
from drl.infrastructure.utils import parser_init

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import MODEL_DEFAULTS

from tqdm import tqdm


    



# ray.init(
#     dashboard_host="0.0.0.0",
#     dashboard_port=40001,
# )

# model_config = MODEL_DEFAULTS.copy()
# model_config["fcnet_hiddens"] = [128, 128]
# model_config["fcnet_activation"] = "tanh"

# trainer = DQNTrainer(env=SingleQBit,
#                      config={
#                             "num_workers": 0,
#                             "num_envs_per_worker": 16,
#                             "num_gpus": 0,
#                             "num_cpus_per_worker": 1,
#                             "noisy": True,
#                             "sigma0": 0.5,
#                             "dueling": True,
#                             "hiddens": [128, 128,]
#                             "gamma": 0.99,
#                             "train_batch_size": 200,
#                             "rollout_fragment_length": 200,
#                             "framework": "torch",
#                             "horizon": 100,
#                             "seed": 42,
#                      })

# trainer = PPOTrainer(env=SingleQBit,
#                      config={
#                          "num_workers": 0,
#                          "num_cpus_for_driver": 8,
#                          "num_envs_per_worker": 4,
#                          "num_gpus": 0,
#                          "num_cpus_per_worker": 1,
#                          "sgd_minibatch_size": 1000,
#                          "lr": 0.0005,
#                          'batch_mode': 'truncate_episodes',
#                          "num_sgd_iter": 10,
#                          "horizon": 1000,
#                          "framework": "torch",
#                          "seed": 42,
#                          "model": model_config,
#                      })

# weights = torch.load("pretrained.pth")
# trainer.set_weights({"default_policy": weights})

# print(trainer.get_weights())


if __name__ == "__main__":
    
    ### ----- PARSING ARGUMENTS ----- ###
    parser = parser_init(argparse.ArgumentParser())
    args = parser.parse_args()
    print(args)
#     for i in tqdm(range(10000)):
#         result = trainer.train()
#         # print(result)
#         if i % 50 == 0:
#             trainer.save()

#     ray.shutdown()