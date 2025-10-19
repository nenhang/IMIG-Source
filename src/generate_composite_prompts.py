import json
import os
import random
import sys
from pathlib import Path
from textwrap import dedent

from openai import OpenAI

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environ_config import DATASET_DIR_2, DEEPSEEK_API_KEY
from src.utils.api_request import check_balance, generate_prompts
from src.utils.json_formatter import dump_formatted_json, parse_json

DEEPSEEK_API_URL = "https://api.deepseek.com"
CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)

EXAMPLE_FORMATS = {
    "human_object": [
        """
        "prompt": "A young man is riding a bright red bicycle down a city street, with his feet on the pedals.",
        "instance": ["man", "bicycle"],
        "instance_prompt": ["A young man", "a bright red bicycle"]
        """,
        """
        "prompt": "A woman is wearing a colorful scarf wrapped loosely around her neck on a chilly day.",
        "instance": ["woman", "scarf"],
        "instance_prompt": ["A woman", "a colorful scarf"]
        """,
        """
        "prompt": "A chef is slicing a ripe tomato on a wooden cutting board with a sharp knife.",
        "instance": ["chef", "tomato"],
        "instance_prompt": ["A chef", "a ripe tomato"]
        """,
        """
        "prompt": "A hiker is holding a walking stick, using it to navigate a rocky path.",
        "instance": ["hiker", "walking stick"],
        "instance_prompt": ["A hiker", "a walking stick"]
        """,
        """
        "prompt": "A child is drawing with a black crayon on a blank sheet of paper.",
        "instance": ["child", "crayon"],
        "instance_prompt": ["A child", "a black crayon"]
        """,
        """
        "prompt": "A person is using a silver garden shovel to dig a hole in a patch of soft soil.",
        "instance": ["person", "garden shovel"],
        "instance_prompt": ["A person", "a silver garden shovel"]
        """,
    ],
    "object_object": [
        """
        "prompt": "A thick book is lying on a soft pillow, with its cover slightly open.",
        "instance": ["book", "pillow"],
        "instance_prompt": ["A thick book", "a soft pillow"]
        """,
        """
        "prompt": "A pair of shiny glasses is resting on a newspaper, with one lens covering a headline.",
        "instance": ["glasses", "newspaper"],
        "instance_prompt": ["A pair of shiny glasses", "a newspaper"]
        """,
        """
        "prompt": "A crumpled napkin is sitting inside a half-empty glass bowl on a wooden table.",
        "instance": ["napkin", "glass bowl"],
        "instance_prompt": ["A crumpled napkin", "a half-empty glass bowl"]
        """,
        """
        "prompt": "A vintage bronze key is stuck inside the keyhole of an old wooden door.",
        "instance": ["key", "wooden door"],
        "instance_prompt": ["A vintage bronze key", "an old wooden door"]
        """,
        """
        "prompt": "A lit white candle is melting onto the wooden surface of a tabletop, forming a wax puddle.",
        "instance": ["candle", "tabletop"],
        "instance_prompt": ["A lit white candle", "the wooden tabletop"]
        """,
        """
        "prompt": "A colorful blanket is draped over the arm of a cozy armchair.",
        "instance": ["blanket", "armchair"],
        "instance_prompt": ["A colorful blanket", "a cozy armchair"]
        """,
    ],
    "animal_object": [
        """
        "prompt": "A fluffy dog is chewing on a large, red toy ball, holding it with its paws.",
        "instance": ["dog", "toy ball"],
        "instance_prompt": ["A fluffy dog", "a large, red toy ball"]
        """,
        """
        "prompt": "A little yellow bird is perched on the rim of an empty birdbath in a garden.",
        "instance": ["bird", "birdbath"],
        "instance_prompt": ["A little yellow bird", "an empty birdbath"]
        """,
        """
        "prompt": "A sleek black cat is sleeping soundly on top of a comfortable, gray pillow.",
        "instance": ["cat", "pillow"],
        "instance_prompt": ["A sleek black cat", "a comfortable, gray pillow"]
        """,
        """
        "prompt": "A small frog is sitting on a large, green lily pad in a calm pond.",
        "instance": ["frog", "lily pad"],
        "instance_prompt": ["A small frog", "a large, green lily pad"]
        """,
        """
        "prompt": "A colorful butterfly is resting on the petal of a bright red flower.",
        "instance": ["butterfly", "flower"],
        "instance_prompt": ["A colorful butterfly", "a bright red flower"]
        """,
        """
        "prompt": "A small turtle is climbing onto a smooth, wet rock in a shallow stream.",
        "instance": ["turtle", "rock"],
        "instance_prompt": ["A small turtle", "a smooth, wet rock"]
        """,
    ],
}


def generate_initial_prompt(num_prompts=40):
    """
    Generates the initial prompt for the LLM, with examples from two random categories.
    """

    # Randomly select two categories
    categories = list(EXAMPLE_FORMATS.keys())
    selected_categories = random.sample(categories, 2)

    # Randomly select one example from each chosen category
    example_1 = random.choice(EXAMPLE_FORMATS[selected_categories[0]]).strip()
    example_2 = random.choice(EXAMPLE_FORMATS[selected_categories[1]]).strip()

    initial_prompt = f"""
Basic requirements:
Generate {num_prompts} scene descriptions in JSON format. Each entry must describe two common subjects with a simple, clear, and direct interaction that results in a visible physical connection or overlap in an image. The prompts should be natural and realistic, using common verbs and prepositions that imply this connection.

Format:
```json
[
    {{
        {example_1}
    }},
    {{
        {example_2}
    }}
...
]```
Specific requirements:
Each entry must have a "prompt", an "instance", and an "instance_prompt" key.
The "instance" list must contain exactly two specific, singular nouns or noun phrases.
The "instance_prompt" list must contain two phrases that describe the subjects with their adjectives and modifiers, matching the ones in the "prompt".
The instance_prompt description must be detailed enough to allow for the generation of a clear, unambiguous instance. Avoid overly generic nouns (e.g., "a net") and instead use more specific descriptions (e.g., "a basketball net").
The prompts should use a mix of dynamic verbs like 'riding', 'sitting', 'carrying' or clear positional prepositions like 'inside of', 'against' to ensure visual overlap.
Use descriptive adjectives to enrich the prompts.
Ensure the generated prompts and subject pairs are highly diverse and do not repeat or closely resemble the provided examples.
Avoid vague terms (sky, ground, water, etc.) and scene-related terms (forest, street, etc.).
Output ONLY the JSON array, nothing else. Do not include any additional text or explanations. Now generate {num_prompts} scene descriptions in the specified JSON format.
"""
    return dedent(initial_prompt).strip()


def llm_prompt_generator(save_path, num_prompts_per_call=40, process_id=0, balance_threshold=15, num_prompts_need=None):
    """
    Main function to generate prompts in a loop until balance is low.
    """
    save_path = save_path[0] if isinstance(save_path, list) else save_path
    balance = check_balance(DEEPSEEK_API_KEY)
    if balance is None:
        print(f"Process {process_id}: Failed to retrieve API balance, stopping generation.")
        return
    else:
        print(f"Process {process_id}: Current API balance: {balance} Yuan")

    batch_count = 0
    num_prompts_generated = 0
    while True:
        initial_prompt = generate_initial_prompt(num_prompts=num_prompts_per_call)
        prompts = generate_prompts(CLIENT, initial_prompt, temperature=1.3)
        json_prompts = parse_json(prompts)

        if not json_prompts:
            print(f"Process {process_id}, batch {batch_count}: No valid prompts generated.")
        else:
            valid_prompts = [
                p
                for p in json_prompts
                if p.get("instance")
                and len(p["instance"]) == 2
                and p.get("instance_prompt")
                and len(p["instance_prompt"]) == 2
            ]

            if not valid_prompts:
                print(f"Process {process_id}, batch {batch_count}: Generated prompts but none had the correct format.")
            else:
                dump_formatted_json(valid_prompts, save_path)
                print(
                    f"Process {process_id}, batch {batch_count}: Generated {len(valid_prompts)} prompts and saved to {save_path}"
                )
                num_prompts_generated += len(valid_prompts)
                if num_prompts_need is not None and num_prompts_generated >= num_prompts_need:
                    print(
                        f"Process {process_id}: Reached the target of {num_prompts_need} prompts, stopping generation."
                    )
                    break

        batch_count += 1

        balance = check_balance(DEEPSEEK_API_KEY)
        if balance is not None and float(balance) < balance_threshold:
            print(f"Process {process_id}: API balance is low ({balance} Yuan), stopping generation.")
            break
        else:
            print(f"Process {process_id}: Current API balance: {balance} Yuan, continuing generation.")


def parallel_llm_prompt_generator(save_path_list, num_prompts_per_call=40, num_processes=4, num_prompts_need=None):
    """
    Runs the prompt generation in parallel using multiple processes.
    """
    from multiprocessing import Pool

    if num_prompts_need is not None:
        num_prompts_need_per_process = num_prompts_need // num_processes
        if num_prompts_need % num_processes != 0:
            num_prompts_need_per_process += 1
    else:
        num_prompts_need_per_process = None

    print(f"Total prompts needed: {num_prompts_need}, Prompts per process: {num_prompts_need_per_process}")

    with Pool(processes=num_processes) as pool:
        pool.starmap(
            llm_prompt_generator,
            [
                (save_path, num_prompts_per_call, i, 15, num_prompts_need_per_process)
                for i, save_path in enumerate(save_path_list)
            ],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate composite prompts using LLM.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--num_prompts_per_call", type=int, default=40, help="Number of prompts per API call.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DATASET_DIR_2,
        help="Directory to save generated prompts.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    save_path_list = []
    cur_prompt_len = 0
    for i in range(args.num_processes):
        save_path_list.append(os.path.join(args.save_dir, f"two_composite_temp_{i + 1}.json"))
        if os.path.exists(save_path_list[-1]):
            with open(save_path_list[-1], "r", encoding="utf-8") as f:
                data = json.load(f)
                cur_prompt_len += len(data)
    print(f"Load {cur_prompt_len} prompts from existing files {save_path_list}")

    print(f"Save paths: {save_path_list}")
    parallel_llm_prompt_generator(
        save_path_list,
        num_prompts_per_call=args.num_prompts_per_call,
        num_processes=args.num_processes,
        num_prompts_need=args.total_prompts - cur_prompt_len if args.total_prompts > cur_prompt_len else 0,
    )

    # gather prompts, assign indices, and save to a single file
    all_prompts = []
    for path in save_path_list:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_prompts.extend(data)
            os.remove(path)
    if len(all_prompts) > args.total_prompts:
        all_prompts = all_prompts[: args.total_prompts]

    # assign indices
    begin_idx = 0
    for i, prompt in enumerate(all_prompts):
        prompt["index"] = begin_idx + i

    final_save_path = os.path.join(args.save_dir, "prompts.json")
    dump_formatted_json(all_prompts, final_save_path)
    print(f"Total {len(all_prompts)} prompts saved to {final_save_path}")
