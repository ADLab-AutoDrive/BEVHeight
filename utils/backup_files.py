""" Simple backup script which just creates the root structure in an other
folder and syncs everything which recursevely lies within one of the source
folders. For files bigger than a threshold they are first gziped."""

import gzip
import os
import shutil
import threading

def size_if_newer(source, target):
    """ If newer it returns size, otherwise it returns False """
    src_stat = os.stat(source)
    try:
        target_ts = os.stat(target).st_mtime
    except FileNotFoundError:
        try:
            target_ts = os.stat(target + '.gz').st_mtime
        except FileNotFoundError:
            target_ts = 0
    # The time difference of one second is necessary since subsecond accuracy
    # of os.st_mtime is striped by copy2
    return src_stat.st_size if (src_stat.st_mtime - target_ts > 1) else False

def threaded_sync_file(source, target):
    size = size_if_newer(source, target)
    if size:
        thread = threading.Thread(target=transfer_file, 
                                  args=(source, target, False))
        thread.start()
        return thread

def transfer_file(source, target, compress):
    """ Either copy or compress and copies the file """
    try:
        if compress:
            with gzip.open(target + '.gz', 'wb') as target_fid:
                with open(source, 'rb') as source_fid:
                    target_fid.writelines(source_fid)
            # print('Compress {}'.format(source))
        else:
            shutil.copy2(source, target)
            # print('Copy {}'.format(source))
    except FileNotFoundError:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        transfer_file(source, target, compress)

def sync_root(target):
    # root: current work dir
    # target: output_dir/backup/
    folder_white_list = ['assets', 'callbacks', 'dataset', 'exps', 'evaluators', 'layers', 'models', 'ops', 'scripts', 'test', 'utils']
    for folder in folder_white_list:
        folder_root = folder
        for path, _, files in os.walk(folder_root):
            for source in files:
                # only backup python files
                source = path + '/' + source
                if source[-3:] == '.py':
                    transfer_file(source, os.path.join(target, source), False)
                    
def backup_codebase(backup_folder):
    sync_root(backup_folder)
    import shutil
    shutil.copy2("setup.py", backup_folder)
    shutil.copy2("train_script.sh", backup_folder)