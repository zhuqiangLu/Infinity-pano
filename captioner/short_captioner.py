from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch
import requests
import json
import PIL
import textwrap
from PIL import Image
import os
import glob
from tqdm import tqdm

def apply_prompt_template(prompt):
    s = (
                '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
                f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
            )
    return s 


# load model
model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)


# setup model
model = model.to('cuda')
model.eval()
tokenizer.padding_side = "left"
tokenizer.eos_token = '<|end|>'


short_caption = dict()
for img_path in tqdm(glob.glob("../data/rename_structured3d/*.png")):
    image_list = []
    image_sizes = []
    # img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    # img = Image.open(requests.get(img_url, stream=True).raw)#.convert('RGB')
    try:
        img = Image.open(img_path)
        query = "describe this 360-degree panorama picture <image>, start with 'A 360-degree panorama picture of'"
        # display.display(Image(filename=fn, width=300))
        image_list.append(image_processor([img], image_aspect_ratio='anyres')["pixel_values"].cuda())
        image_sizes.append(img.size)
        inputs = {
            "pixel_values": [image_list]
        }
        # for query in sample['question']:
        prompt = apply_prompt_template(query)
        language_inputs = tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        # To cuda
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[name] = value.cuda()
        generated_text = model.generate(**inputs, image_size=[image_sizes],
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        temperature=0.05,
                                        do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1,
                                        )
        prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
        short_caption[img_path] = prediction
    except:
        continue


# Save short captions to a file
import json
with open('./structured3d_short_captions.json', 'w') as f:
    json.dump(short_caption, f, indent=4)
