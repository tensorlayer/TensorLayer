import argparse
from tensorlayer.cli import trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tl')
    subparsers = parser.add_subparsers(dest='cmd')
    train_parser = subparsers.add_parser('train', help='train a model using multiple local GPUs and CPUs.')
    trainer.build_arg_parser(train_parser)
    args = parser.parse_args()
    if args.cmd == 'train':
        trainer.main(args)
