import wandb
import math

wandb.login()

sweep_config = {
  'method': 'random'
}

# metric = {
#   'name': 'micro f1 score',
#   'goal': 'maximize'
# }

# sweep_config['metric'] = metric

parameters_dict = {
    'entity_flag': {
        'values': [True, False]
        },
    'preprocessing_cmb': {
        'values': ['3', '2', '2 3', '1', '1 3', '1 2', '1 2 3', '0', '0 3', '0 2', '0 2 3', '0 1', '0 1 3', '0 1 2', '0 1 2 3']
        },
    'mecab_flag': {
        'values': [True, False]
        },
    'aeda_flag': {
        'values': [True, False]
        },
    'augmentation_flag': {
        'values': [True, False]
        },
    'train_batch_size': {
        'values': [2, 4, 8, 16, 32, 64, 128, 256]
      }
}

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'lr': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 1e-6,
        'max': 1e-3
      },
    # 'train_batch_size': {
    #     # integers between 32 and 256
    #     # with evenly-distributed logarithms 
    #     'distribution': 'q_log_uniform',
    #     'q': 1,
    #     'min': math.log(8),
    #     'max': math.log(256),
    #   }
})