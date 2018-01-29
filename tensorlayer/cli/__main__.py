import argparse
from tensorlayer.experimental import local_trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tl')
    subparsers = parser.add_subparsers(dest='cmd')
    train_parser = subparsers.add_parser('train', help='train a model')
    local_trainer.build_arg_parser(train_parser)
    args = parser.parse_args()
    if args.cmd == 'train':
        local_trainer.main(args)
