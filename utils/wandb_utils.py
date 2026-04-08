import os

import wandb


def add_wandb_args(parser):
    parser.add_argument(
        '--wandb_mode',
        default=os.getenv('WANDB_MODE', 'disabled'),
        choices=['disabled', 'offline', 'online'],
        help='Weights & Biases mode: disabled, offline, or online.',
    )
    parser.add_argument(
        '--no_wandb',
        action='store_true',
        default=False,
        help='Disable Weights & Biases logging.',
    )


def resolve_wandb_mode(args):
    if getattr(args, 'no_wandb', False):
        return 'disabled'
    return getattr(args, 'wandb_mode', 'disabled')


def init_wandb_run(*, args, project, run_name, config):
    return wandb.init(
        project=project,
        name=run_name,
        id=run_name,
        mode=resolve_wandb_mode(args),
        resume='allow',
        config=config,
    )