import sys
import argparse
from app import app
from app import route
from waitress import serve

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # model_choices = ['vgg16', 'multitask']

    parser.add_argument('model', type=str)
    parser.add_argument('cut_point', type=int)
    parser.add_argument('next_cut_point', nargs='+', type=int)
    parser.add_argument('--is-first', action='store_true')
    parser.add_argument('--is-last', action='store_true')
    parser.set_defaults(is_first=False)
    parser.set_defaults(is_last=False)

    args = parser.parse_args()

    print(args.next_cut_point)
    
    route.init(model_name=args.model, cut_point=args.cut_point, next_cut_point=args.next_cut_point, is_first=args.is_first, is_last=args.is_last)
    serve(app, host='0.0.0.0', port=5000)