python task_training.py device.cuda_visible_devices="5" \
        output_dir=test/resnet18_224/resnet18_linear/imagenet_80_100 \
        task.data.classes_used_to_train="0-20" \
        task.data.group_used_to_train="0-20"\
        task.param.data_root=param_data/test/resnet18_linear/imagenet_80_100/data.pt \
        task.train_layer=['"fc.weight","fc.bias"']\
        process_title=test

python task_training.py device.cuda_visible_devices="3" \
        output_dir=test/resnet18_224/resnet18_linear/imagenet_0_20 \
        task.data.classes_used_to_train="80-100" \
        task.data.group_used_to_train="80-100"\
        task.param.data_root=test/resnet18_linear/imagenet_0_20/data.pt \
        task.train_layer=['"fc.weight","fc.bias"']\
        > ./test/test4.out &

python task_training.py device.cuda_visible_devices="4" \
        output_dir=test/resnet18_224/resnet18_linear/imagenet_0_20 \
        task.data.classes_used_to_train="80-100" \
        task.data.group_used_to_train="80-100"\
        task.param.data_root=test/resnet18_linear/imagenet_0_20/data.pt \
        task.train_layer=['"fc.weight","fc.bias"']\
        > ./test/test5.out

python task_training.py device.cuda_visible_devices="5" \
        output_dir=test/resnet18_224/resnet18_linear/imagenet_0_20 \
        task.data.classes_used_to_train="80-100" \
        task.data.group_used_to_train="80-100"\
        task.param.data_root=test/resnet18_linear/imagenet_0_20/data.pt \
        task.train_layer=['"fc.weight","fc.bias"']\
        > ./test/test6.out

python task_training.py device.cuda_visible_devices="6" \
        output_dir=test/resnet18_224/resnet18_linear/imagenet_0_20 \
        task.data.classes_used_to_train="80-100" \
        task.data.group_used_to_train="80-100"\
        task.param.data_root=test/resnet18_linear/imagenet_0_20/data.pt \
        task.train_layer=['"fc.weight","fc.bias"']\
        > ./test/test7.out


# python task_training.py device.cuda_visible_devices="3" \
#         output_dir=test/resnet18_224/resnet18_linear/imagenet_0_20 \
#         task.data.classes_used_to_train="0-20" \
#         task.data.group_used_to_train="0-20"\
#         task.param.data_root=test/resnet18_224/resnet18_linear/imagenet_0_20/data.pt \
#         task.train_layer=['"fc.weight","fc.bias", "layer4.1.conv2.weight", "layer4.1.bn2.weight","layer4.1.bn2.bias"']\
#         > ./test/test2.out