from llava.model.builder import load_pretrained_model
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import disable_torch_init
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict
import transformers
import re
from PIL import Image
from datasets import load_dataset
import os
import requests
from io import BytesIO
import json


model_path = 'neulab/CulturalPangea-7B'
model_name = 'CulturalPangea-7B-qwen'
args = {"multimodal": True}
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path=model_path, model_base=None, model_name=model_name,attn_implementation="sdpa" , **args)



def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    input_ids = []
    source = sources
    if roles[source[0]["from"]] != roles["human"]: source = source[1:]
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1: _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None: _input_id = tokenizer(role).input_ids + nl_tokens
            else: _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
    input_ids.append(input_id)
    return torch.tensor(input_ids, dtype=torch.long)

def generate_output(prompt, image=None, do_sample=False, temperature=0, top_p=0.5, num_beams=1, max_new_tokens=1024):
    image_tensors = []
    prompt = "<image>\n" + prompt
    # image can be a path to a local file or a PIL image
    if isinstance(image, str):
        image = Image.open(image)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    image_tensors.append(image_tensor.half().cuda())
    input_ids = preprocess_qwen([{'from': 'human', 'value': prompt},{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    return outputs


CACHE_DIR = "/network/scratch/m/maltaism/datasets/image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_image(example):
    filename = example["Image ID"]
    path = os.path.join(CACHE_DIR, filename)

    if not os.path.exists(path):
        response = requests.get(example["URL"], timeout=10)
        response.raise_for_status()  # catch 4xx/5xx errors early

        # Validate it's actually an image before caching
        try:
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception:
            raise ValueError(f"URL did not return a valid image: {example['URL']}")

        with open(path, "wb") as f:
            f.write(response.content)

        return img  # already loaded, no need to re-open

    return Image.open(path).convert("RGB")


def main():
    original_prompt = "What cultural significance does the following image have?"
    
    ds = load_dataset("Multimedia-SMU/seeingculture-benchmark")
    shuffled_dataset = ds.shuffle(seed=42)
    output_file = "RESULTS/CulturalPangea-7B-qwen/seeing_culture_question_baseline_results_new.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counter = 0
    results = []  # accumulate in memory

    for example in shuffled_dataset["test"]:
        try:
            image = load_image(example)
            grounded_prompt = f"What cultural significance does the following image have in {example['Country']}?"
            question_prompt = example["Question"]
            prediction = generate_output(question_prompt, image=image)
            print("Image ID:", example["Image ID"], "| Prediction:", prediction)

            results.append({
                "Image ID": example["Image ID"],
                "URL": example["URL"],
                "Prediction": prediction,
                "country": example["Country"],
                "category": example["Category"],
                "question": example["Question"],
                "answer": example["Rationale"],
            })
            counter += 1
            print("Processed:", example["Image ID"], "| Total processed:", counter)

            # Flush to disk every 50 examples instead of every iteration
            if counter % 50 == 0:
                with open(output_file, "a") as f:
                    for r in results:
                        json.dump(r, f)
                        f.write("\n")
                results = []  # clear buffer

        except Exception as e:
            print("Skipping:", example["Image ID"], "|", e)
            continue

    # Final flush for any remaining results
    if results:
        with open(output_file, "a") as f:
            for r in results:
                json.dump(r, f)
                f.write("\n")


if __name__ == "__main__":
    main()