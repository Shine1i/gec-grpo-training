import argparse
import json
def main(args):
    for mode in 'bin cat1 cat2 cat3'.split(' '):
        for model in 'bert-base-cased roberta-base xlnet-base-cased google/electra-base-discriminator bert-large-cased roberta-large xlnet-large-cased google/electra-large-discriminator'.split(' '):
            best_f05 = 0
            best_path = ''
            for seed in '10 11 12'.split():
                path = f'models/{mode}/{model}/seed{seed}/log.json'
                try:
                    result = json.load(open(path))
                except FileNotFoundError:
                    continue
                for ep, r in result.items():
                    if best_f05 < r['valid_log']['cls_report']['1']['f05-score']:
                        best_f05 = max(
                            best_f05,
                            r['valid_log']['cls_report']['1']['f05-score']
                        )
                        best_path = path + f' [{ep}]'
            if best_path != '':
                print(best_path)
                print(best_f05)
            
                    

def get_parser():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)