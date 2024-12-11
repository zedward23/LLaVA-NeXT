# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model, lxr_load_llava_next_ov
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
base = "llava_qwen"
checkpoint = "checkpoints/onevision/llava-onevision-qwen2-0.5b-si"
#lora_checkpoint = "checkpoints/onevision/lora_llava-onevision-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-0.5b-si-ov_stage_am9"

device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(
    model_path=checkpoint, 
    model_base=None, #review the error message with Marcel.  is it the padding? why is the answer so big?
    model_name=base,
    device_map=device_map
)
model.eval()

image_path = "images/banana3.png"  # Replace with the path to your local image
image = None

if "https" in image_path:
    image = Image.open(requests.get(image_path, stream=True).raw)
else:
    image = Image.open(image_path)

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
question = DEFAULT_IMAGE_TOKEN + "\n  is this a banana?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

#import the keywords thing
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
#look


from transformers import TextStreamer
streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True
)
cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=True,
    temperature=1,
    max_new_tokens=256,
    streamer=streamer,
    stopping_criteria=[stopping_criteria],
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

