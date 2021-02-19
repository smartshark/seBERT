from pathlib import Path
import argparse
import glob

import nltk

parser = argparse.ArgumentParser()

parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to be converted into the input format expected by BERT; \
          accept '**/*.txt' type of patterns if enclosed in quotes",
)

parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved",
)

args = parser.parse_args()

files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)

def sent_tokenize(files, out):
    for _file in files:
        lines = []
        processed = []
        with open(_file, encoding='utf-8') as reader:
            lines = reader.readlines()
        for line in lines:
            if not line.isspace():
                processed.append('\n'.join(nltk.sent_tokenize(line)))
        with open(Path(out) / Path(_file).name, 'w', encoding='utf-8') as writer:
            writer.write('\n\n'.join(processed))
    
sent_tokenize(files, args.out)