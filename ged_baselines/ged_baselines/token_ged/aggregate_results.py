import argparse
import glob
import os
import pprint

def read_score(path):
    content = open(path).read().rstrip().split('\n')
    def extract_prf(lines):
        assert len(lines) == 3
        p = float(lines[0][len('Precision: '):])
        r = float(lines[1][len('recall: '):])
        f = float(lines[2][len('f0.5: '):])
        return (p, r, f)
    return {
        'path': path,
        'bea-valid': extract_prf(content[2:5]),
        'fce-test': extract_prf(content[13:16]),
        'conll14-0': extract_prf(content[24:27]),
        'conll14-1': extract_prf(content[35:38]),
    }

def read_score_macro(path):
    content = open(path).read().rstrip().split('\n')
    content = content[5:]
    def extract_prf(lines):
        assert len(lines) == 3
        p = float(lines[0][len('Precision: '):])
        r = float(lines[1][len('recall: '):])
        f = float(lines[2][len('f0.5: '):])
        return (p, r, f)
    
    return {
        'path': path,
        'bea-valid': extract_prf(content[2:5]),
        'fce-test': extract_prf(content[13:16]),
        'conll14-0': extract_prf(content[24:27]),
        'conll14-1': extract_prf(content[35:38]),
    }

def to_latex(score, row_name='bert-base-cased'):
    def three_to_one(temp):
        return f'({round(temp[0]*100, 2)} / {round(temp[1]*100, 2)} / {round(temp[2]*100, 2)})'
    print(f'|{row_name}|', three_to_one(score['bea-valid']), '|', three_to_one(score['fce-test']), '|', three_to_one(score['conll14-0']), '|', three_to_one(score['conll14-1']), '|')

def main(args):
    paths = glob.glob(os.path.join(args.dir, 'seed*/best/score.txt'))
    ind = 0
    scores = []
    for path in paths:
        score = read_score(path)
        scores.append(score)
    for i, s in enumerate(scores):
        if scores[ind]['bea-valid'][2] < s['bea-valid'][2]:
            ind = i
    print(scores[ind]['path'])
    to_latex(scores[ind], 'bert-large-cased')
            
    ind = 0
    scores = []
    for path in paths:
        score = read_score_macro(path)
        scores.append(score)
    for i, s in enumerate(scores):
        if scores[ind]['bea-valid'][2] < s['bea-valid'][2]:
            ind = i
    print(scores[ind]['path'])
    to_latex(scores[ind], 'bert-large-cased')
        
        
        

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)