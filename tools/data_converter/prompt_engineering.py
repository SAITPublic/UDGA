import numpy as np
import torch
import clip
import argparse


cams = ['FRONT LEFT', 'FRONT', 'FRONT RIGHT', 'BACK LEFT', 'BACK', 'BACK RIGHT']
focal_lengths = ['large focal length', 'short focal length']
backgrounds = ['day', 'night', 'rainy', 'foggy', 'highway', 'urban']

# prompt_templates = ['A {0} scene captured from the {1} {2} camera of an autonomous vehicle.',
#             'A {0} scene recorded by the {1} {2} view of an self-driving car.',
#             'A {0} scene obtained through the {1} {2} perspective of an self-driving car.',
#             'A {0} scene taken via the {1} {2} angle of an autonomous vehicle.',
#             'A {0} scene acquired from the {1} {2} viewpoint of an autonomous car.',
#             'A {0} scene captured through the {1} {2} viewpoint of an autonomous vehicle.',
#             'A {0} scene secured through the {1} {2} angle of an autonomous car.',
#             'A {0} scene taken with the {1} {2} perspective of an self-driving car.',
#             'A {0} scene secured through the {1} {2} view of an autonomous vehicle.',
#             'A {0} scene observed through the {1} {2} camera of an self-driving car.',
# ]

prompt_templates = [
                    'A photo captured from the {} camera of an autonomous vehicle.',
                    'A image recorded by the {} view of an self-driving car.',
                    'A shot obtained through the {} perspective of an self-driving car.',
                    'A scene taken via the {} angle of an autonomous vehicle.',
                    'A data acquired from the {} viewpoint of an autonomous car.',
                    'A shot captured through the {} viewpoint of an autonomous vehicle.',
                    'A image secured through the {} angle of an autonomous car.',
                    'A photh taken with the {} perspective of an self-driving car.',
                    'A picture secured through the {} view of an autonomous vehicle.',
                    'A image observed through the {} camera of an self-driving car.',
                    ]



def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engeering script')
    # parser.add_argument('--model', default='RN50', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='clip model name')
    parser.add_argument('--model', default='ckpts/ViT-B-16.pt')
    parser.add_argument('--class-set', default=['nuscenes'], nargs='+',
        choices=['lyft', 'nuscenes'],
        help='the set of class names')
    parser.add_argument('--no-prompt-eng', action='store_true', help='disable prompt engineering')

    args = parser.parse_args()
    return args

# def zeroshot_classifier(model_name, classnames, templates):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load(model_name, device=device)
#     with torch.no_grad():
#         zeroshot_weights_cam = []
#         for cam in cams:
#             zeroshot_weights_focal = []
#             for focal in focal_lengths:
#                 zeroshot_weights_back = []
#                 for back in backgrounds:
#                     texts = [template.format(back, focal, cam) for template in templates] #format with class
#                     texts = clip.tokenize(texts).cuda() #tokenize
#                     class_embeddings = model.encode_text(texts) #embed with text encoder
#                     class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
#                     class_embedding = class_embeddings.mean(dim=0)
#                     class_embedding /= class_embedding.norm()
#                     zeroshot_weights_back.append(class_embedding)
#                 zeroshot_weights_focal.append(torch.stack(zeroshot_weights_back, dim=0).cuda())
#             zeroshot_weights_cam.append(torch.stack(zeroshot_weights_focal, dim=0).cuda())
#         zeroshot_weights_cam = torch.stack(zeroshot_weights_cam, dim=0).cuda()
#     return zeroshot_weights_cam

def zeroshot_classifier(model_name, classnames, templates):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    with torch.no_grad():
        zeroshot_weights_cam = []
        for cam in cams:
            texts = [template.format(cam) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights_cam.append(class_embedding)
        zeroshot_weights_cam = torch.stack(zeroshot_weights_cam, dim=0).cuda()
    return zeroshot_weights_cam


if __name__ == '__main__':
    args = parse_args()

    print('Hello World!')

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16'}
    # zeroshot_weights = zeroshot_classifier(name_mapping[args.model], classes, prompt_templates)
    zeroshot_weights = zeroshot_classifier(args.model, args.class_set, prompt_templates)
    zeroshot_weights = zeroshot_weights.squeeze().float()
    # A, B, C, D = zeroshot_weights.shape
    # zeroshot_weights = zeroshot_weights.view(A,B*C,D).permute(1,0,2)
    torch.save(zeroshot_weights, f'ckpts/clip/camera_extrinsic_clip_text.pth')
