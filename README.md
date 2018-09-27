# Torch-Scope

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/tensorboard-wrapper/badge/?version=latest)](http://tensorboard-wrapper.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/torch-scope)](https://pepy.tech/project/torch-scope)
[![PyPI version](https://badge.fury.io/py/torch-scope.svg)](https://badge.fury.io/py/torch-scope)

A Toolkit for training pytorch models, which has three features:

- Tracking environments, dependency, implementations and checkpoints;
- Providing a logger wrapper with two handlers (to ```std``` and ```file```);
- Supporting automatic device selection;
- Providing a tensorboard wrapper;
- Providing a spreadsheet writer to automatically summarizing notes and results;

We are in an early-release beta. Expect some adventures and rough edges.

## Quick Links

- [Installation](#installation)
- [Usage](#usage)

## Installation

To install via pypi:
```
pip install torch-scope
```

To build from source:
```
pip install git+https://github.com/LiyuanLucasLiu/Torch-Scope
```
or
```
git clone https://github.com/LiyuanLucasLiu/Torch-Scope.git
cd Torch-Scope
python setup.py install
```

## Usage

An example is provided as below, please read the doc for a detailed api explaination.

* set up the git in the server & add all source file to the git
* use tensorboard to track the model stats (tensorboard --logdir PATH/log/ --port ####)

```
from torch_scope import wrapper
...

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', type=str, ...)
    parser.add_argument('--name', type=str, ...)
    parser.add_argument('--gpu', type=str, ...)
    ...
    args = parser.parse_args()

    pw = wrapper(os.path.join(args.checkpoint_path, args.name), name = args.log_dir, enable_git_track = False)
    # Or if the current folder is binded with git, you can turn on the git tracking
    # pw = wrapper(os.path.join(args.checkpoint_path, args.name), name = args.log_dir, enable_git_track = True)

    gpu_index = tbw.auto_device() if 'auto' == args.gpu else int(args.gpu)
    device = torch.device("cuda:" + str(gpu_index) if gpu_index >= 0 else "cpu")

    pw.save_configue(args) # dump the config to config.json

    pw.set_level('info') # or 'debug', etc.

    pw.info(str(args)) # would be plotted to std & file if level is 'info' or lower

    ...

    batch_index = 0

    for index in range(epoch):

    	...

    	for instance in ... :

    		loss = ...

    		tot_loss += loss.detach()
    		loss.backward()

    		if batch_index % ... = 0:
    			pw.add_loss_vs_batch({'loss': tot_loss / ..., ...}, batch_index, False)
    			pw.add_model_parameter_stats(model, batch_index, save=True)
    			optimizer.step()
    			pw.add_model_update_stats(model, batch_index)
    			tot_loss = 0
    		else:
    			optimizer.step()

    		batch_index += 1

    	dev_score = ...
    	pw.add_loss_vs_batch({'dev_score': dev_score, ...}, index, True)

    	if dev_score > best_score:
    		pw.save_checkpoint(model, optimizer, is_best = True)
    		best_score = dev_score
    	else:
    		pw.save_checkpoint(model, optimizer, is_best = False)
```

## Advanced Usage

### Auto Device

### Git Tracking

### Spreadsheet Logging

Share the spreadsheet with the following account ```torch-scope@torch-scope.iam.gserviceaccount.com```. And access the table with its name. 
