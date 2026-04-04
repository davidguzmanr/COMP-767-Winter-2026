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

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
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


def format_for_dpo(example: dict, country: str, rejected_answer: str) -> dict:
    """Convert a CulturalGround example + rejected response to TRL DPO format."""
    human_text, gpt_text = _parse_conversation(example)

    prompt = [
        {"role": "system", "content": build_system_message(country)},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": human_text},
            ],
        },
    ]
    return {
        "prompt":   prompt,
        "chosen":   [{"role": "assistant", "content": gpt_text}],
        "rejected": [{"role": "assistant", "content": rejected_answer}],
        "images":   [example["image"]],
    }


# ---------------------------------------------------------------------------
# Rejected-response generation (DPO only)
# ---------------------------------------------------------------------------

def generate_rejected_responses(
    examples: list[dict],
    model_name_or_path: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> list[str]:
    """
    Generate rejected responses from a reference model prompted *without*
    cultural conditioning, producing the generic/Western-centric answers that
    serve as the 'rejected' side of preference pairs.
    """
    processor = AutoProcessor.from_pretrained(model_name_or_path, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    rejected = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        conversations = []
        images = []
        for ex in batch:
            human_text, _ = _parse_conversation(ex)
            conversations.append([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": human_text},
                ]}
            ])
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
            rejected.append(decoded.strip())

    return rejected


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

    if args.output_format == "sft":
        processed = raw.map(
            lambda ex: format_for_sft(ex, country),
            remove_columns=raw.column_names,
            num_proc=args.dataset_num_proc,
            desc=f"SFT formatting ({country})",
        )

    else:  # dpo
        print(f"  Generating rejected responses with {args.reference_model_name_or_path} ...")
        examples = list(raw)
        rejected_responses = generate_rejected_responses(
            examples,
            model_name_or_path=args.reference_model_name_or_path,
            batch_size=args.dpo_rejected_batch_size,
            max_new_tokens=args.dpo_rejected_max_new_tokens,
        )
        formatted = [
            format_for_dpo(ex, country, rej)
            for ex, rej in zip(examples, rejected_responses)
        ]
        processed = Dataset.from_list(formatted)

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
