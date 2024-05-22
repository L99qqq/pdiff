name = 'imagenet_resnet18_linear'

def create_task_order(start_class, end_class):
    task_each_gpu = 3

    class_incre=20
    gap = 1

    gpu=[0,1,2,3,4,5,6,7]
    cur_gpu=0

    cnt = 0
    
    with open("script_train/task_orders.sh", "w") as file:
        for i in range(start_class,end_class,gap):
            cnt += 1
            file.write('nohup python task_training.py device.cuda_visible_devices="{}" \
                        process_title=modelzoo_imagenet{}_{}  output_dir=outputs/modelzoo_imagenet_20/{}_{} \
                        task.data.classes_used_to_train="{}-{}" task.data.group_used_to_train="{}-{}" \
                        task.param.data_root=param_data/modelzoo_imagenet_20/{}_{}/data.pt >./task_nohup_out/modelzoo_imagenet_20/{}_{}.out 2>&1 & \n\n'.format( \
                            gpu[cur_gpu],i,i+class_incre,i,i+class_incre,i,i+class_incre,i,i+class_incre,i,i+class_incre,i,i+class_incre))
            file.write('pid{}=$!\nsleep 140\n'.format(cnt))
            if cnt == task_each_gpu * len(gpu):
            # if(cur_gpu==len(gpu)-1):
                file.write('wait')
                for j in range(cnt):
                    file.write('$pid{} '.format(j+1))
                file.write('\n')
                cnt = 0
            cur_gpu=(cur_gpu+1)%len(gpu)



original_start = 854
final_start = 880
# final_start = 880

create_task_order(start_class=original_start, end_class=final_start+1)
