#model settings
model = dict(

    type = 'EncoderDecoder3D',
    backbone = dict(
        type = 'PonintNet',
        in_channels = 3,
        channels = [64,128,1024],
        global_feat = False,
        feature_transform =True,
    ),
    decode_head = dict(
        in_channels = 1088,
        channels =[512,256,128],
    ),

)