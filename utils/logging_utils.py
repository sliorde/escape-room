from os import walk, makedirs
from os.path import dirname, abspath, relpath, join, splitext, basename
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import pickle

def save_to_zip(output_dir):
    zipf = ZipFile(join(output_dir,'files.zip'), 'w', ZIP_DEFLATED)
    root = dirname(dirname(abspath(__file__)))
    for dir, subdirs, files in walk(root):
        for file in files:
            if file.endswith('.py'):
                zipf.write(join(dir, file),arcname=relpath(join(dir, file), root))
    zipf.close()

def get_output_dir(name):
    name = splitext(basename(name))[0]
    t = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_dir = join('checkpoints', name, t)
    makedirs(output_dir,exist_ok=True)
    return output_dir

def save_params(output_dir,**kwargs):
    with open(join(output_dir,'params.pickle'),'wb') as f:
        pickle.dump(kwargs,f)