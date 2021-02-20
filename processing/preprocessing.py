import argparse
import glob
import re
from bs4 import BeautifulSoup
import fasttext
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--files",
    default=None,
    metavar="path",
    type=str,
    required=True,
    help="The files to be preprocessed; \
          accept '**/*.txt' type of patterns if enclosed in quotes"
)

parser.add_argument(
    "--out",
    default="./",
    type=str,
    help="Path to the output directory, where the files will be saved"
)

parser.add_argument(
    "--hash",
    default=True,
    type=bool,
    help="Defines whether to remove hashes."
)

parser.add_argument(
    "--html",
    default=False,
    type=bool,
    help="Should be set to true if the files contain html tags."
)

parser.add_argument(
    "--md",
    default=False,
    type=bool,
    help="Should be set to true if the files are markdown formatted."
)

parser.add_argument(
    "--mentions",
    default=False,
    type=bool,
    help="Defines whether to remove user mentions. @user"
)

parser.add_argument(
    "--commit",
    default=False,
    type=bool,
    help="Defines whether to remove commit message specific ids and signed-off-by."
)

parser.add_argument(
    "--jira",
    default=False,
    type=bool,
    help="Defines whether to remove jira specific formatting and code blocks"
)

args = parser.parse_args()

# expected input format is .txt, one document per line
files = glob.glob(args.files)
if not files:
    print(f"File does not exist: {args.files}")
    exit(1)

# fasttext model
pretrained = "lid.176.bin"
pretrained_model = fasttext.load_model(pretrained)

# user mentions
user_mentions = re.compile(r'(?<![@\w])@(\w{1,38})')
   
# markdown specific regex
md_quote = re.compile(r'((?:^|>)[^<]*?)>')
md_formatting = re.compile(r'#{2,6}|__|\*{2,3}|~~')

# regex's for markdown [description](url)
url_desc = re.compile(r'\[([^[]+?)\]\(.+?\)')
url = re.compile(r'(?<=\])\(.+?\)')

# regex's for special hashes
former_commit_id = re.compile(r'former-commit-id:\s[0-9a-f]{40}')
change_id = re.compile(r'change-Id:\si[0-9a-f]{40}')
svn_git_id = re.compile(r'git-svn-id:\s[0-9a-f]{40}@\d{1,10}\s[0-9a-z]{8}(-[0-9a-z]{4}){3}-[0-9a-z]{12}')
signed_off_by = re.compile(r'signed-off-by:\s([^(\s<)]+\s){1,5}<[^\s]+>')

# formatting specific to jira issues
jira_code = re.compile(r'\{code.*\}.+?\{code.*\}')
jira_noformat = re.compile(r'\{noformat.*\}.+?\{noformat.*\}')

''' removes control characters '''
def remove_control(s):
    s = s.replace('\n', ' ')
    s = s.replace('\t', ' ')
    s = s.replace('\r', ' ')
    s = s.replace('\\n', ' ')
    s = s.replace('\\t', ' ')
    s = s.replace('\\r', ' ')
    return s

''' determines if the document is english '''
def is_english(s):
    return (pretrained_model.predict(s, k=1)[0][0][len('__label__'):] == 'en')

''' replace user mentions with special [USER] token '''
def replace_mentions(s):
    return re.sub(user_mentions, ' [USER] ', s)

''' removes markdown code blocks and extracts text '''
def remove_html(s):
    soup = BeautifulSoup(s, 'lxml')
    for c in soup.find_all('code'):
        c.replace_with(' [CODE] ')
    return soup.get_text()

''' replaces markdown url formatting with description and url seperated '''
def clean_urls(s):
    url_descs = re.findall(url_desc, s)
    if len(url_descs) > 0:
        urls = re.findall(url, s)
        for i, desc in enumerate(url_descs):
            s = s.replace('[' + desc + ']', ' ' + desc + ' ', 1)
            s = s.replace(urls[i], urls[i][1:-1])
    return s

''' removes markdown formatting '''
def remove_md(s):
    s = re.sub(md_formatting, ' ', s)
    s = re.sub(md_quote, ' ', s)
    s = re.sub(r'```.+?```', ' [CODE] ', s, 0, re.DOTALL)
    return clean_urls(s)

''' removes jira specific formatting '''
def remove_jira_formatting(s):
    s = re.sub(jira_noformat, ' ', s)
    s = s.replace('{quote}', ' ')
    return re.sub(jira_code, ' [CODE] ', s)

''' removes github specific hashes '''
def remove_gh_special_hashes(s):
    s = re.sub(signed_off_by, ' ', s)
    s = re.sub(change_id, ' ', s)
    s = re.sub(svn_git_id, ' ', s)
    return re.sub(former_commit_id, ' ', s)

''' normalizes quotation marks '''
def normalize_qm(s):
    s = s.replace('\\"', '"')
    s = s.replace('“', '"')
    return s.replace('”', '"')

''' normalizes whitespaces '''
def cleanup_whitespaces(s):
    return ' '.join(s.split())

''' determines whether the string contains alphabetic characters '''
def has_alpha(s):
    return (sum(c.isalpha() for c in s) >= 1)

''' determines whether the string is a hexadecimal number'''
def is_hex(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        return False

''' replaces hash values with special [HASH] token '''
def replace_hashes(s):
    matches = [word for word in s.split(' ') if word.isalnum() and len(word) >= 7 and is_hex(word) and has_alpha(word)]
    for match in matches:
        s = s.replace(match, ' [HASH] ', 1)
    return s

''' special tokens and needed punctations sometimes get corrupted
by other preprocessing steps '''
def repair(s):
    s = s.replace(' CODE] ', ' [CODE] ')
    s = s.replace(' HASH] ', ' [HASH] ')
    s = s.replace(' USER] ', ' [USER] ')
    s = s.replace(' [CODE ', ' [CODE] ')
    s = s.replace(' [HASH ', ' [HASH] ')
    s = s.replace(' [USER ', ' [USER] ')
    s = s.replace('[ USER ]', ' [USER] ')
    s = s.replace('[ CODE ]', ' [CODE] ')
    s = s.replace('[ HASH ]', ' [HASH] ')
    s = s.replace(' . ', '. ')
    return s.replace(' , ', ', ')

def preprocess(files, out):
    for _file in files:
        lines = []
        processed = []
        with open(_file, encoding='utf-8') as reader:
            lines = reader.readlines()
            for line in lines:
                if line != '\n':
                    # remove cases
                    doc = line.lower()
                    # html extraction
                    if args.html:
                        doc = remove_html(doc)
                    # markdown extraction
                    if args.md:
                        doc = remove_md(doc)
                    # remove control characters
                    doc = remove_control(doc)
                    # normalize quotation marks
                    doc = normalize_qm(doc)
                    # replace user mentions                        
                    if args.mentions:
                        doc = replace_mentions(doc)
                    # replace hashes
                    if args.hash:
                        doc = replace_hashes(doc)
                    # special commit message preprocessing
                    if args.commit:
                        doc = remove_gh_special_hashes(doc)
                    # remove jira formatting
                    if args.jira:
                        doc = remove_jira_formatting(doc)
                    # fix corrupt special tokens
                    doc = repair(doc)
                    # clean up whitespaces
                    doc = cleanup_whitespaces(doc)                
                    if len(doc) > 0 and is_english(doc):
                        processed.append(doc)
        with open(Path(out) / Path(_file).name, 'w', encoding='utf-8') as writer:
            lines = []
            for line in processed:
                doc = ' '.join(line.split())
                if not doc.isspace(): 
                    lines.append(doc)
            writer.write('\n\n'.join(lines))

preprocess(files, args.out)
