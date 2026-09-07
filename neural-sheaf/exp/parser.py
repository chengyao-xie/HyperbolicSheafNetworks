import argparse

from exp.config import FORMAL_DEFAULTS


def get_parser():
    parser = argparse.ArgumentParser()
    parser.set_defaults(**FORMAL_DEFAULTS)
    parser.add_argument('--dataset', default='abide')
    parser.add_argument(
        '--construction_method', choices=['causality', 'pearson'],
        default='causality')
    parser.add_argument('--simplex_order', type=int, choices=[1, 2], default=2)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--preprocess_workers', type=int, default=0)
    return parser
