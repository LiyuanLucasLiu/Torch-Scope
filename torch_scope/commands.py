"""
.. module:: check
    :synopsis: check
.. moduleauthor:: Liyuan Liu
"""
import os
import git
import json
import argparse
import logging
from torch_scope import basic_wrapper

logger = logging.getLogger(__name__)

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
    logger.info('Now the implementation has been restored for {}'.format(args.checkpoint_path))
    logger.info('Please checkout to the master branch after finished.')
    
def run():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Commands', metavar='')

    subcommands = {
            "checkout": checkout()
    }

    for name, subcommand in subcommands.items():
        subparser = subcommand.add_subparser(name, subparsers)

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if 'func' in dir(args):
        args.func(args)
    else:
        parser.print_help()
