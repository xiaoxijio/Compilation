# _base_ = ['../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
#           '../../../mmdetection3d/configs/_base_/default_runtime.py']


_base_ = ['../../../configs/_base_/datasets/nus-3d.py',
          '../../../configs/_base_/default_runtime.py']


plugin = True
# plugin_dir = 'projects/mmdet3d_plugin/'
# plugin_dir = 'projects/FlashOCC/flashocc'
custom_imports = dict(imports=['FlashOCC_Board'])
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    # 'x': [-40, 40, 0.4],b  # 200 grids
    'x': [-25.6, 25.6, 0.4],  # 128 grids
    # 'y': [-40, 40, 0.4], # 200 grids
    'y': [-25.6, 25.6, 0.4],  # 128 grids
    'z': [-1, 5.4, 6.4],  # height = 6.4 meters
    'depth': [1.0, 45.0, 1.0],  # 44 depth values
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 64

model = dict(
    type='BEVDetOCC',
    img_backbone=dict(
        type='EfficientNetB0HZ',
        pretrained=None,
    ),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[112, 1280],
        out_channels=256,
        num_outs=1,
        start_level=0,
        out_ids=[0], 
        upsample_cfg=dict(mode='bilinear', align_corners=False)
        ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=256,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=True,
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=128,
        channels_factor=1),
    occ_head=dict(
        type='BEVOCCHead2D',
        in_dim=128,
        out_dim=128,
        Dz=16,
        use_mask=True,
        num_classes=18,
        use_predicter=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0
        ),
    )
)

# Data
dataset_type = 'NuScenesDatasetOccpancy'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_depth', 'voxel_semantics',
                                'mask_lidar', 'mask_camera'])
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'])
        ])
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2occ_opt-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=8,
    # samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2occ_opt-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

# # Optimizer
# optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
# optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=200,
#     warmup_ratio=0.001,
#     step=[24, ])
# runner = dict(type='EpochBasedRunner', max_epochs=24)

# # Starting Evaluation...
# # 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:16<00:00, 363.00it/s]
# # ===> per class IoU of 6019 samples:
# # ===> others - IoU = 5.47
# # ===> barrier - IoU = 32.34
# # ===> bicycle - IoU = 2.17
# # ===> bus - IoU = 31.71
# # ===> car - IoU = 41.53
# # ===> construction_vehicle - IoU = 17.68
# # ===> motorcycle - IoU = 6.38
# # ===> pedestrian - IoU = 12.11
# # ===> traffic_cone - IoU = 10.17
# # ===> trailer - IoU = 22.21
# # ===> truck - IoU = 27.81
# # ===> driveable_surface - IoU = 82.55
# # ===> other_flat - IoU = 41.34
# # ===> sidewalk - IoU = 50.71
# # ===> terrain - IoU = 54.22
# # ===> manmade - IoU = 32.24
# # ===> vegetation - IoU = 28.86
# # ===> mIoU of 6019 samples: 29.38
# # {'mIoU': array([0.055, 0.323, 0.022, 0.317, 0.415, 0.177, 0.064, 0.121, 0.102,
# #        0.222, 0.278, 0.826, 0.413, 0.507, 0.542, 0.322, 0.289, 0.901])}


# Optimizer
optimizer = dict(type='AdamW', lr=4e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[30, 50, 60])
runner = dict(type='EpochBasedRunner', max_epochs=60)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]

load_from = "ckpts/BEVDet/bevdet-r50-cbgs.pth"
fp16 = dict(loss_scale='dynamic')
evaluation = dict(interval=1, start=100, pipeline=test_pipeline)
checkpoint_config = dict(interval=1, max_keep_ckpts=60)


# Starting Evaluation...
# 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6019/6019 [00:34<00:00, 172.51it/s]
# ===> per class IoU of 6019 samples:
# ===> others - IoU = 7.59
# ===> barrier - IoU = 40.5
# ===> bicycle - IoU = 8.97
# ===> bus - IoU = 38.04
# ===> car - IoU = 47.26
# ===> construction_vehicle - IoU = 19.07
# ===> motorcycle - IoU = 11.63
# ===> pedestrian - IoU = 16.18
# ===> traffic_cone - IoU = 14.2
# ===> trailer - IoU = 30.68
# ===> truck - IoU = 32.86
# ===> driveable_surface - IoU = 84.11
# ===> other_flat - IoU = 44.81
# ===> sidewalk - IoU = 53.91
# ===> terrain - IoU = 56.59
# ===> manmade - IoU = 37.2
# ===> vegetation - IoU = 33.26
# ===> mIoU of 6019 samples: 33.93
# {'mIoU': array([0.076, 0.405, 0.09 , 0.38 , 0.473, 0.191, 0.116, 0.162, 0.142,
#        0.307, 0.329, 0.841, 0.448, 0.539, 0.566, 0.372, 0.333, 0.904])}
