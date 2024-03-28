python3 ./rtdetr_pytorch/tools/inference_and_save_results.py \
    --config "./rtdetr_pytorch/configs/rtdetr/alarad_c1_rtdetr_r101vd_6x.yml" \
    --resume "./rtdetr_pytorch/alarad_str_best_weight_c3/checkpoint0119.pth" \
    --data ../ALARAD_Strawberry_Dataset_final/ALARAD_Strawberry_1060_c1/images/test \
    --save_dir ./rtdetr_pytorch/output
    --dir_name predictions \
    --save_img true --save_txt true
