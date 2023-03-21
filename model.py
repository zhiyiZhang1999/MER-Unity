'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''

from torch import nn

from models import multimodalcnn, multimodalcnn_lstm

import os, torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 2"

def generate_model(opt):
    assert opt.model in ['multimodalcnn']

    if opt.model == 'multimodalcnn':
        # 8, softhard, 15, 'EfficientFace_Trained_on_AffectNet7.pth.tar', 1
        model = multimodalcnn.MultiModalCNN(opt.n_classes, fusion = opt.fusion, seq_length = opt.sample_duration, pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.model == 'multimodalcnn_lstm':
        # 8, softhard, 15, 'EfficientFace_Trained_on_AffectNet7.pth.tar', 1
        model = multimodalcnn_lstm.MultiModalCNN(opt.n_classes, fusion=opt.fusion, seq_length=opt.sample_duration,
                                            pretr_ef=opt.pretrain_path, num_heads=opt.num_heads)

    if opt.device != 'cpu':
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model = nn.DataParallel(model, device_ids=[0, 1])
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    
    return model, model.parameters()
