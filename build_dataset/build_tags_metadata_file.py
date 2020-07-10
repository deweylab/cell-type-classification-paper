import json
from optparse import OptionParser

def main():
    usage = ""
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    exp_set_f = args[0]
    annot_f = args[1]
    out_f = options.out_file

    with open(annot_f, 'r') as f:
        annot = json.load(f)
    with open(exp_set_f, 'r') as f:
        exps = json.load(f)['experiments']

    exp_to_tags = {
        exp: annot[exp]['tags']
        for exp in exps
    }
    with open(out_f, 'w') as f:
        json.dump(exp_to_tags, f, indent=4)           

if __name__ == '__main__':
    main()
