import os
import tempfile

import torch


def save(obj, dir, file_name):
    tmp = tempfile.NamedTemporaryFile(delete=False, dir=os.path.expanduser(dir))

    try:
        torch.save(obj, tmp.file)
    except BaseException:
        tmp.close()
        os.remove(tmp.name)
        raise
    else:
        tmp.close()
        os.rename(tmp.name, os.path.join(dir, file_name))
