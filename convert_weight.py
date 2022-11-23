#convert paddle weight to .pth
#usage: Place the official weight file "model.pdparams" in the root directory and run the file. It will generate "pretrained_model.pth"

import paddle.fluid as fluid
import torch
import numpy as np

from nets.uhrnet import UHRnet
from nets.uhrnet_training import weights_init

path = 'model.pdparams'   # Modify according to the actual
model   = UHRnet(num_classes=19, backbone = 'UHRNet_W18_Small')   # Modify "backbone" as needed: UHRNet_W18_Small/UHRNet_W48

weights_init(model)
state = fluid.io.load_program_state(path)

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dict      = model.state_dict()

model_keys = model_dict.keys()
trans_dict = {}
for model_key in model_keys:

    if 'last' not in model_key:
        model_key = model_key.replace('.0.','.')
        model_key = model_key.replace('.1.','.')
    model_key_split = model_key.split('.')
    pp_key = None
    for word in model_key_split:
        if 'list' not in word:
            if pp_key is not None:
                pp_key = pp_key + '.' + word
            else:
                pp_key = word
    pp_key = pp_key.replace('.bn.','._batch_norm.')
    pp_key = pp_key.replace('.running_mean','._mean')
    pp_key = pp_key.replace('.running_var','._variance')
    pp_key = pp_key.replace('.conv.','._conv.')
    pp_key = pp_key.replace('la1.bottleneck_block_list.bb_layer2','la1.bb_layer2')
    trans_dict[pp_key] = model_key


load_key, no_load_key, temp_dict = [], [], {}
for k, v in state.items():
    if k in trans_dict.keys():
        if trans_dict[k] in model_dict.keys() and np.shape(model_dict[trans_dict[k]]) == np.shape(v):
            temp_dict[trans_dict[k]] = torch.FloatTensor(v)
            load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)
#------------------------------------------------------#
#   显示没有匹配上的Key
#------------------------------------------------------#
print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

torch.save(model.state_dict(), 'pretrained_model.pth')
