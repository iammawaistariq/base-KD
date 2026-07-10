import sys
sys.path.insert(0, 'src')

print('sys.path[0]=', sys.path[0])

try:
    from vim_kd.data import build
    print('imported build from', build.__file__)
except Exception as e:
    print('import error', e)
    raise

import torchvision
from torchvision import transforms as T
print('torchvision', torchvision.__version__)
print('has RandAugment in torchvision.transforms:', hasattr(T, 'RandAugment'))

# Exercise _build_transforms
try:
    t_train = build._build_transforms('cifar10', 224, train=True)
    print('train transforms built:', t_train)
except Exception as e:
    print('error building train transforms:', repr(e))

try:
    t_eval = build._build_transforms('cifar10', 224, train=False)
    print('eval transforms built:', t_eval)
except Exception as e:
    print('error building eval transforms:', repr(e))
