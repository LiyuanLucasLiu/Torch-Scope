# tensorboard-wrapper

[![Documentation Status](https://readthedocs.org/projects/tensorboard-wrapper/badge/?version=latest)](http://tensorboard-wrapper.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Toolkit for training models based on pytorch, which has three features:

1. Tracking environments, dependency, implementations and checkpoints;
2. Providing a logger wrapper with two handlers (to ```std``` and ```file```);
3. Providing a tensorboard wrapper

## Quick Links

- [Installation](#installation)
- [Usage](#usage)

## Installation

To build from source:
```
git clone https://github.com/LiyuanLucasLiu/tensorboard-wrapper.git
cd tensorboard-wrapper
python setup.py install
```

## Usage

An example is provided as below, please read the doc for a detailed api explaination.

* set up the git in the server & add all source file to the git
* use tensorboard to track the model stats (tensorboard --logdir PATH/log/ --port ####)

```
from tbwrapper import wrapper
...

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, ...)
    parser.add_argument('--name', type=str, ...)
    ...
    args = parser.parse_args()

    wp = wrapper(os.path.join(args.checkpoint_path, args.name), name = args.log_dir, enable_git_track = True)

    wp.save_configue(args) # dump the config to config.json

    wp.set_level('info') # or 'debug', etc.
    logger = wp.get_logger()

    logger.info(str(args)) # would be plotted to std & file if level is 'info' or lower

    ...

    batch_index = 0

    for index in range(epoch):

    	...

    	for instance in ... :

    		loss = ...

    		tot_loss += loss.detach()
    		loss.backward()

    		if batch_index % ... = 0:
    			wp.add_loss_vs_batch({'loss': tot_loss / ..., ...}, batch_index, False)
    			wp.add_model_parameter_stats(model, batch_index, save=True)
    			optimizer.step()
    			wp.add_model_update_stats(model, batch_index)
    			tot_loss = 0
    		else:
    			optimizer.step()

    		batch_index += 1

    	dev_score = ...
    	wp.add_loss_vs_batch({'dev_score': dev_score, ...}, index, True)

    	if dev_score > best_score:
    		wp.save_checkpoint(model, optimizer, is_best = True)
    		best_score = dev_score
    	else:
    		wp.save_checkpoint(model, optimizer, is_best = False)
```