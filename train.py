#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = ArgumentParser()
    # args
    parser.add_argument('mode', type=str, choices=['normal', 'extra'], default='normal')
    parser.add_argument('stage_or_nepoch', type=int, help='If mode=normal, this specifies the target stage (inclusive) to train until. If mode=extra, this specifies the number of epochs to train for.')
    # optional args
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-n', '--new', action='store_true', help='Create a model from scratch.')
    group.add_argument('-l', '--load', type=str, help='Load from a specified save.')
    group.add_argument('-L', '--load-latest', type=str, help='Load the latest save in a directory.')
    parser.add_argument('-s', '--save', type=str, help='Custom save path.')
    return parser.parse_args()

def find_latest_save(dir):
    path = Path(dir)
    # get all save_paths
    save_paths = [p for p in path.glob('save_?_*_*') if p.is_dir()]
    # find the latest
    save_paths = sorted(save_paths, key=lambda p: datetime.strptime(p.name[7:], f'%d-%m-%Y_%H-%M'))
    save_paths.insert(0, None)
    return save_paths[-1]

def check_dir(path):
    if not Path(path).is_dir():
        print(f'Directory "{path}" does not exist!')

def main():
    args = parse_args()

    # get model
    from gan.gan import WGAN
    if args.new:
        model = WGAN({ 'stage' : 1, 'during_fadein' : False })
    elif args.load:
        check_dir(args.load)
        model = WGAN(checkpoint_path=args.load)
    elif args.load_latest:
        check_dir(args.load_latest)
        latest_path = find_latest_save(args.load_latest)
        model = WGAN(checkpoint_path=latest_path)
    else:
        print('At least one of -n, -l, or -L must be specified.')
        import sys
        sys.exit(-1)
    
    # perform normal / extra training
    if args.mode == 'normal':
        stage = args.stage_or_nepoch
        print(f'[Normal] target stage {stage} \
                model loaded at stage {model.stage} \
                {"with" if model.config["during_fadein"] else "without"} fadein.')
        # perform remaining epochs if applicable
        if not args.new:
            model.fit_remaining_epochs()
        while model.stage < stage:
            model.one_growth_cycle()
    else: # extra
        initial_epoch = model.config['epoch'] + 1
        print(f'[Extra] num epoch {args.stage_or_nepoch}, \
                starting epoch {initial_epoch + 1}; \
                model loaded at stage {model.stage} \
                {"with" if model.config["during_fadein"] else "without"} fadein.')
        model.fit_n_epochs(initial_epoch + args.stage_or_nepoch, initial_epoch=initial_epoch)

    # save model
    model.save(checkpoint_path=args.save)

if __name__ == '__main__':
    main()