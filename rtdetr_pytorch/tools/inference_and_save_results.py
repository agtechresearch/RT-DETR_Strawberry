import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import time
import torch
import torch.nn as nn 
from torchvision import transforms, ops
from PIL import Image, ImageDraw, ImageFont
from src.core import YAMLConfig

def create_directory(directory, name):
    new_folder_name = name
    counter = 1
    while os.path.exists(os.path.join(directory, new_folder_name)):
        new_folder_name = f"{name}{counter}"
        counter += 1
    new_path = os.path.join(directory, new_folder_name)
    os.makedirs(new_path)
    print(f"{new_folder_name} created")
    return new_path

def process_image(image_path, transform, model, label_colors, classes, thrh, fin_dest, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image = Image.open(image_path)
    image_name = os.path.basename(image_path)
    image_width, image_height = image.size

    data = transform(image).to(device)
    size = torch.tensor([[640, 640]]).to(device)
    prediction = model(data, size) 
    labels, boxes, scores = prediction

    ratio_width = image_width / 640
    ratio_height = image_height / 640
    original_box_coordinates = boxes.clone() 
    original_box_coordinates[0][:, [0, 2]] *= ratio_width
    original_box_coordinates[0][:, [1, 3]] *= ratio_height

    result_list = []
    for i in range(len(labels[0])):
        score = float(scores[0][i])
        if score >= thrh:
            label = int(labels[0][i]) - 1
            box = original_box_coordinates[0][i].tolist()
            result_list.append([label, box, score])

    save_path = os.path.join(fin_dest, image_name)

    if args.save_img:
        draw = ImageDraw.Draw(image)
        for label, box, score in result_list:
            x1, y1, x2, y2 = box
            color = label_colors[label]
            txt_color = (255, 255, 255)
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)  
            label_text = classes[label]
            font = ImageFont.load_default()
            w, h = font.getsize(label_text)
            outside = box[1] - h >= 0
            draw.rectangle(
                (box[0], box[1] - h if outside else box[1], box[0] + w + 1, box[1] + 1 if outside else box[1] + h + 1),
                fill=color,
            )
            draw.text((box[0], box[1] - h if outside else box[1]), label_text, fill=txt_color, font=font)
        image.save(save_path)
        print('Prediction result saved at ', save_path)

    if args.save_txt:
        save_txt_path = save_path.split('png')[0] + 'txt'
        with open(save_txt_path, "w") as file:
            for label, box, _ in result_list:  
                box_str = ' '.join(str(coord) for coord in box)
                line = f"{label} {box_str}\n"
                file.write(line)

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = YAMLConfig(args.config, resume=args.resume)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  
        transforms.ToTensor(),           
    ])

    label_colors = ['#FF3838', '#00C2FF', '#FF701F', '#8A2BE2', '#CFD231', '#48F90A', '#FFC0CB', '#3DDB86', '#1A9334',
                    '#00D4BB']
    classes = ['Bud','Flower','Receptacle','Early fruit','White fruit','50% maturity','80% maturity']

    thrh = 0.47 #threshold

    fin_dest = create_directory(args.save_dir, args.dir_name)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device) 
        state = checkpoint.get('ema', {}).get('module', checkpoint.get('model'))
    else:
        raise AttributeError('only support resume to load model.state_dict by now.')
    
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            print(self.postprocessor.deploy_mode)
            #print("init-----------------")
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            #print(outputs)
            
            return self.postprocessor(outputs, orig_target_sizes) 
    model = Model().to(device)
    model_load_time = time.time()
    model_load_time_end = time.time()

    image_folder_path = args.data 
    image_paths = [os.path.join(image_folder_path, filename) for filename in os.listdir(image_folder_path)
                    if filename.lower().endswith((".jpg", ".jpeg", ".png"))]

    time_for_save = 0
    inference_start = time.time()
    for image_path in image_paths:
        process_image(image_path, transform, model, label_colors, classes, thrh, fin_dest, args)

    inference_end = time.time()
    model_loading_time = (model_load_time_end - model_load_time) * 1000
    process_time_total = (inference_end - inference_start) * 1000
    inference_time_except_save = process_time_total - time_for_save
    inference_per_image = inference_time_except_save / len(image_paths)

    print("model_loading: {:.2f}ms, inference_time_except_save: {:.2f}ms, total {:.2f}ms".format(model_loading_time, 
                                                                                                inference_time_except_save, 
                                                                                                process_time_total))
    print("inference per image: {:.2f}ms".format(inference_per_image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--data', '-data', type=str)
    parser.add_argument('--save_dir','-sdir', type=str, default='/home/cv_task/RT-DETR/rtdetr_pytorch/output')
    parser.add_argument('--dir_name','-dirn', type=str, default='predictions')
    parser.add_argument('--save_img','-simg', type=bool, help='Flag to save images')
    parser.add_argument('--save_txt','-stxt', type=bool, help='Flag to save txt files')
    
    args = parser.parse_args()
    main(args)
