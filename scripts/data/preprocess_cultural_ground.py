"""
Dataset preprocessing for CulturalGround → TRL-compatible formats.

CulturalGround (neulab/CulturalGround, re-hosted at davidguzmanr/CulturalGround)
stores data per-country as separate HuggingFace configs.  Each example has:

    {
        "id":            "Q27797552_None",
        "conversations": [
            {"from": "human", "value": '<image>\\n"What is ...?"'},
            {"from": "gpt",   "value": '"The municipality shown ..."'},
        ],
        "image":    <PIL.Image>,
        "language": "en",
    }

This script converts one or more country configs into either:

  SFT format (TRL SFTTrainer):
    {
        "messages": [
            {"role": "system",    "content": "..."},   # cultural conditioning
            {"role": "user",      "content": [{"type": "image"},
                                              {"type": "text", "text": "..."}]},
            {"role": "assistant", "content": "..."},
        ],
        "images": [<PIL.Image>],
    }

  DPO format (TRL DPOTrainer):
    {
        "prompt":   [{"role": "system", ...}, {"role": "user", ...}],
        "chosen":   [{"role": "assistant", "content": "..."}],
        "rejected": [{"role": "assistant", "content": "..."}],
        "images":   [<PIL.Image>],
    }

Usage (SFT, single country):
    python scripts/data/preprocess_cultural_ground.py \\
        --countries mexico \\
        --output_format sft \\
        --push_to_hub \\
        --repo_id <your-hf-id>/CulturalGround-sft

Usage (SFT, all countries):
    python scripts/data/preprocess_cultural_ground.py \\
        --countries all \\
        --output_format sft \\
        --push_to_hub \\
        --repo_id <your-hf-id>/CulturalGround-sft

Usage (DPO):
    python scripts/data/preprocess_cultural_ground.py \\
        --countries mexico india japan \\
        --output_format dpo \\
        --reference_model_name_or_path <sft-checkpoint> \\
        --push_to_hub \\
        --repo_id <your-hf-id>/CulturalGround-dpo
"""

import re
from dataclasses import dataclass, field
from io import BytesIO

import PIL.Image
import torch
from datasets import Dataset, DatasetDict, Image as HFImage, Sequence, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, HfArgumentParser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO = "davidguzmanr/CulturalGround"

ALL_COUNTRIES = [
    "bangladesh",
    "brazil",
    "bulgaria",
    "china",
    "czechia",
    "egypt",
    "ethiopia",
    "france",
    "germany",
    "greece",
    "india",
    "indonesia",
    "iran",
    "ireland",
    "israel",
    "italy",
    "japan",
    "kenya",
    "malaysia",
    "mexico",
    "mongolia",
    "netherlands",
    "nigeria",
    "norway",
    "pakistan",
    "poland",
    "portugal",
    "romania",
    "russia",
    "rwanda",
    "saudi_arabia",
    "singapore",
    "south_korea",
    "spain",
    "sri_lanka",
    "taiwan",
    "tanzania",
    "thailand",
    "turkey",
    "ukraine",
    "united_kingdom",
    "vietnam",
]

# System prompt templates (same as preprocess_datasets.py for consistency)
SYSTEM_MESSAGE_TEMPLATE = (
    "You are a helpful assistant with expertise in {culture} culture. "
    "When answering questions, consider the cultural context of {country}."
)


def build_system_message(country: str) -> str:
    # Normalize config name (e.g. "saudi_arabia" → "Saudi Arabia") for readability
    readable = country.replace("_", " ").title()
    return SYSTEM_MESSAGE_TEMPLATE.format(culture=readable, country=readable)


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------

_IMAGE_TOKEN_RE = re.compile(r"<image>\s*", re.IGNORECASE)


def _parse_conversation(example: dict) -> tuple[str, str]:
    """
    Extract (human_text, gpt_text) from a CulturalGround conversations list.

    The human value typically looks like:
        '<image>\\n"What is the name of the municipality shown ...?"'

    We strip the <image> token and surrounding whitespace; the rest becomes
    the text part of the user message.
    """
    human_text = gpt_text = ""
    for turn in example.get("conversations", []):
        if turn["from"] == "human":
            human_text = _IMAGE_TOKEN_RE.sub("", turn["value"]).strip()
        elif turn["from"] == "gpt":
            gpt_text = turn["value"].strip()
    return human_text, gpt_text


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------

def _is_valid_image(example: dict) -> bool:
    """Return False for examples whose image bytes cannot be decoded by PIL.

    The image column must be in non-decoded form (a dict with a 'bytes' key)
    so that datasets does not raise before we get a chance to catch the error.
    """
    try:
        img = example["image"]
        if img is None:
            return False
        raw_bytes = img.get("bytes") if isinstance(img, dict) else None
        if raw_bytes is None:
            return False
        PIL.Image.open(BytesIO(raw_bytes)).verify()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------

def format_for_sft(example: dict, country: str) -> dict:
    """Convert a single CulturalGround example to TRL SFT conversational format."""
    human_text, gpt_text = _parse_conversation(example)

    messages = [
        {"role": "system", "content": build_system_message(country)},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": human_text},
            ],
        },
        {"role": "assistant", "content": gpt_text},
    ]
    return {"messages": messages, "images": [example["image"]]}


def format_for_dpo(example: dict, country: str, chosen_answer: str, rejected_answer: str) -> dict:
    """Convert a CulturalGround example + chosen/rejected responses to TRL DPO format."""
    human_text, _ = _parse_conversation(example)

    # content must always be a list so PyArrow infers a consistent schema.
    prompt = [
        {"role": "system", "content": [{"type": "text", "text": build_system_message(country)}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "text": None},
                {"type": "text", "text": human_text},
            ],
        },
    ]
    return {
        "prompt":   prompt,
        "chosen":   [{"role": "assistant", "content": [{"type": "text", "text": chosen_answer}]}],
        "rejected": [{"role": "assistant", "content": [{"type": "text", "text": rejected_answer}]}],
        "images":   [example["image"]],
    }


# ---------------------------------------------------------------------------
# Rejected-response generation (DPO only)
# ---------------------------------------------------------------------------

def generate_responses(
    examples: list[dict],
    model_name_or_path: str,
    system_message: str | None = None,
    reference_answers: list[str] | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: str = "cuda",
    desc: str = "Generating responses",
    processor=None,
    model=None,
) -> list[str]:
    """
    Generate responses from a reference model.

    If ``system_message`` is None the model is prompted without cultural
    conditioning (produces generic/Western-centric answers → rejected side).
    If ``system_message`` is provided it is prepended as a system turn
    (produces culturally-conditioned answers → chosen side).

    If ``reference_answers`` is provided, each ground-truth answer is appended
    to the user message so the model generates in its own style while staying
    faithful to the correct answer.

    Pass ``processor`` and ``model`` to reuse already-loaded objects and avoid
    loading weights twice when generating both chosen and rejected responses.
    """
    if processor is None:
        processor = AutoProcessor.from_pretrained(model_name_or_path, do_image_splitting=False)
    if model is None:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch.bfloat16,
            device_map=device,
        )
        model.eval()

    responses = []
    for i in tqdm(range(0, len(examples), batch_size), desc=desc):
        batch = examples[i : i + batch_size]
        batch_refs = reference_answers[i : i + batch_size] if reference_answers is not None else None
        conversations = []
        images = []
        for j, ex in enumerate(batch):
            human_text, _ = _parse_conversation(ex)
            if batch_refs is not None:
                user_text = f"{human_text}\n\nReference answer: {batch_refs[j]}"
            else:
                user_text = human_text
            turns = []
            if system_message is not None:
                turns.append({"role": "system", "content": system_message})
            turns.append({"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ]})
            conversations.append(turns)
            images.append(ex["image"])

        texts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in conversations
        ]
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        for ids in output_ids:
            decoded = processor.decode(ids[input_len:], skip_special_tokens=True)
            responses.append(decoded.strip())

    return responses


# ---------------------------------------------------------------------------
# Script arguments & main
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    countries: list[str] = field(
        default_factory=lambda: ["mexico"],
        metadata={
            "help": (
                "Country config(s) to process. Pass 'all' to process every country, "
                "or one/more country names (e.g. --countries mexico india japan)."
            )
        },
    )
    dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to preprocess."},
    )
    output_format: str = field(
        default="sft",
        metadata={"help": "Output format. One of: 'sft', 'dpo'."},
    )
    merge_countries: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, concatenate all country splits into a single dataset before "
                "pushing/saving.  If False, push each country as a separate config."
            )
        },
    )
    reference_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Reference model for generating DPO rejected responses. Required when output_format='dpo'."},
    )
    dpo_rejected_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for generating DPO rejected responses."},
    )
    dpo_rejected_max_new_tokens: int = field(
        default=128,
        metadata={"help": "Max new tokens when generating DPO rejected responses."},
    )
    dpo_holdout_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "If set, reserve the last N valid examples of each country for DPO and use "
                "the rest for SFT, preventing overlap between the two datasets. "
                "When output_format='dpo' the last N examples are kept; "
                "when output_format='sft' the last N examples are excluded."
            )
        },
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "If set, truncate each country split to this many examples (useful for testing)."},
    )
    regenerate_chosen: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, generate the chosen response from the reference model with cultural "
                "conditioning instead of using the dataset's ground-truth answer. "
                "Both chosen and rejected will then share the model's writing style."
            )
        },
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of workers for dataset .map() calls."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the processed dataset to the Hugging Face Hub."},
    )
    repo_id: str | None = field(
        default=None,
        metadata={"help": "HuggingFace Hub repo ID to push the dataset to."},
    )


def process_country(
    country: str,
    args: ScriptArguments,
) -> Dataset:
    """Load and convert one country config to the requested output format."""
    print(f"\n=== {country} ===")
    raw = load_dataset(HF_REPO, name=country, split=args.dataset_split, trust_remote_code=True)
    print(f"  Loaded {len(raw)} examples.")

    n_before = len(raw)
    # Cast to non-decoded so PIL errors are catchable inside _is_valid_image.
    raw_nodecode = raw.cast_column("image", HFImage(decode=False))
    valid_indices = [
        i for i, ex in enumerate(raw_nodecode)
        if _is_valid_image(ex)
    ]
    raw = raw.select(valid_indices)
    if args.dpo_holdout_size is not None:
        n = min(args.dpo_holdout_size, len(raw))
        if args.output_format == "dpo":
            raw = raw.select(range(len(raw) - n, len(raw)))
        else:  # sft
            raw = raw.select(range(len(raw) - n))
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))
    n_skipped = n_before - len(raw)
    if n_skipped:
        print(f"  Skipped {n_skipped} examples with undecodable images.")

    if args.output_format == "sft":
        processed = raw.map(
            lambda ex: format_for_sft(ex, country),
            remove_columns=raw.column_names,
            num_proc=args.dataset_num_proc,
            desc=f"SFT formatting ({country})",
        )

    else:  # dpo
        print(f"  Generating responses with {args.reference_model_name_or_path} ...")
        # Decoded PIL images for the generation model
        examples_decoded = list(raw)
        # Non-decoded byte dicts for Arrow-compatible dataset construction
        examples_raw = list(raw.cast_column("image", HFImage(decode=False)))

        # Load once and reuse for both passes when regenerate_chosen is set.
        shared_processor = AutoProcessor.from_pretrained(
            args.reference_model_name_or_path, do_image_splitting=False
        )
        shared_model = AutoModelForImageTextToText.from_pretrained(
            args.reference_model_name_or_path, dtype=torch.bfloat16, device_map="cuda"
        )
        shared_model.eval()

        rejected_responses = generate_responses(
            examples_decoded,
            model_name_or_path=args.reference_model_name_or_path,
            system_message=None,
            batch_size=args.dpo_rejected_batch_size,
            max_new_tokens=args.dpo_rejected_max_new_tokens,
            desc="Generating rejected responses",
            processor=shared_processor,
            model=shared_model,
        )

        if args.regenerate_chosen:
            ground_truths = [_parse_conversation(ex)[1] for ex in examples_decoded]
            chosen_responses = generate_responses(
                examples_decoded,
                model_name_or_path=args.reference_model_name_or_path,
                system_message=build_system_message(country),
                reference_answers=ground_truths,
                batch_size=args.dpo_rejected_batch_size,
                max_new_tokens=args.dpo_rejected_max_new_tokens,
                desc="Generating chosen responses",
                processor=shared_processor,
                model=shared_model,
            )
        else:
            # Use ground-truth answers from the dataset directly.
            chosen_responses = [_parse_conversation(ex)[1] for ex in examples_decoded]

        formatted = [
            format_for_dpo(ex, country, chosen, rej)
            for ex, chosen, rej in zip(examples_raw, chosen_responses, rejected_responses)
        ]
        # Pop images out before from_list to avoid Arrow schema inference issues
        # with the image struct, then add back as a properly typed column.
        images = [f.pop("images") for f in formatted]
        processed = Dataset.from_list(formatted)
        processed = processed.add_column("images", images).cast_column("images", Sequence(HFImage()))

    print(f"  → {len(processed)} processed examples.")
    return processed


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if args.output_format not in ("sft", "dpo"):
        raise ValueError(f"Unknown output_format '{args.output_format}'. Choose 'sft' or 'dpo'.")

    if args.output_format == "dpo" and args.reference_model_name_or_path is None:
        raise ValueError("--reference_model_name_or_path is required when --output_format dpo")

    if args.push_to_hub and args.repo_id is None:
        raise ValueError("--repo_id is required when --push_to_hub is set")

    countries = ALL_COUNTRIES if args.countries == ["all"] else args.countries
    invalid = [c for c in countries if c not in ALL_COUNTRIES]
    if invalid:
        raise ValueError(f"Unknown country config(s): {invalid}. Valid options: {ALL_COUNTRIES}")

    country_datasets: dict[str, Dataset] = {}
    for country in countries:
        country_datasets[country] = process_country(country, args)

    if args.merge_countries:
        merged = concatenate_datasets(list(country_datasets.values()))
        print(f"\nTotal examples after merging {len(countries)} countries: {len(merged)}")
        print(merged[0])

        if args.push_to_hub:
            merged.push_to_hub(args.repo_id)
            print(f"Merged dataset pushed to {args.repo_id}")
    else:
        # Push each country as a separate config
        for country, ds in country_datasets.items():
            print(ds[0])
            if args.push_to_hub:
                ds.push_to_hub(args.repo_id, config_name=country)
                print(f"  Pushed {country} → {args.repo_id} (config={country})")


if __name__ == "__main__":
    main()
