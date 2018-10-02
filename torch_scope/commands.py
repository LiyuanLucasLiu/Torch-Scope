"""
.. module:: check
    :synopsis: check
.. moduleauthor:: Liyuan Liu
"""
import os
import git
import json
import argparse

from torch_scope import basic_wrapper

class checkout():
    def add_subparser(self, name, parser):
        subparser = parser.add_parser(name, description="checkout the implementation for a checkpoint", help='Checkout a Checkpoint')
        subparser.add_argument('-c', '--checkpoint-path', required=True, type=str, help='Path to the checkpoint root')
        subparser.set_defaults(func=checkout_checkpoint)

        return subparser

def checkout_checkpoint(args):
	config = basic_wrapper.restore_configue(args.checkpoint_path, 'environ.json')

	assert('GIT HEAD COMMIT' in config)

	repo = git.Repo(config['PATH'], search_parent_directories=True)
	repo.git.checkout(config['GIT HEAD COMMIT'])
	