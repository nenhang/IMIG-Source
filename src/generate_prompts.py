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

from environ_config import DATASET_DIR, DEEPSEEK_API_KEY
from src.utils.api_request import check_balance, generate_prompts
from src.utils.json_formatter import dump_formatted_json, parse_json

# DeepSeek API client setup
DEEPSEEK_API_URL = "https://api.deepseek.com"
CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)

# style templates
STYLE_TEMPLATES = {
    "realistic": {
        "prefix": "A photorealistic scene showing",
    },
    "anime": {
        "prefix": "An anime-style illustration featuring",
    },
    "watercolor": {
        "prefix": "A watercolor painting depicting",
    },
    "cyberpunk": {
        "prefix": "A neon-lit cyberpunk scene with",
    },
}

# Generation rules
GENERATION_RULES = {
    "instance": {
        "count": "**more than 3** main instances with clear visual features. make sure at least **one human instance** is included!!!",
        "size": "each instance must large enough to be identifiable, especially humans (or human faces)",
        "id": "distinguishable through clothing, accessories or unique markings. human instances should be described in detail, such as facial features, expressions, or clothing.",
        "others": "add interactions or relationships between instances (especially humans) to some of the prompts",
        "constraint": "when you're listing nouns, you should follow these rules: "
        "don't list background environment names that can't be easily detected, like 'sky', 'ground', 'water', etc. "
        "don't list those instance which are not visually identifiable, like 'sunlight', 'shadow', 'wind', etc. "
        "and don't list the instance that belongs to another instance, like a human's glasses or the features of a human's face."
        "However, it's okay to mention them in the prompt text",
    },
    "background": {
        "elements": "some secondary objects providing scene context",
        "description": "suggest background , either briefly or in detail",
        "constraint": "background elements should not dominate main instances",
    },
}


GENERATION_RULES_2 = {
    "instance": {
        "count": "**more than 5** main instances with clear visual features. make sure at least **two human instance** is included!!!",
        "size": "each instance must be large enough to be identifiable, especially **human faces, which should be prominent and clearly visible**.",  # Modified
        "id": "distinguishable through clothing, accessories or unique markings. at least one of the human instances should be described in detail, such as **detailed facial features, expressions, or clothing**.",  # Modified
        "others": "add interactions or relationships between instances (especially humans) to the prompts",
        "constraint": "when you're listing nouns, you should follow these rules: "
        "don't list background environment names that can't be easily detected, like 'sky', 'ground', 'water', etc. "
        "don't list those instance which are not visually identifiable, like 'sunlight', 'shadow', 'wind', etc. "
        "and don't list the instance that belongs to another instance, like a human's glasses or the features of a human's face."
        "However, it's okay to mention them in the prompt text",
    },
    "background": {
        "elements": "some secondary objects providing scene context",
        "description": "suggest background , either briefly or in detail",
        "constraint": "background elements should not dominate main instances",
        "spatial_cues": "**include spatial cues** like 'on the left' ('to the left'), 'on the right' ('to the right'), 'above', 'below', 'in front of', 'behind' to describe the relative positions of instances or background elements.",  # New rule
    },
}

EXAMPLE_FORMATS = {
    "realistic": [
        {
            "prompt": "A photorealistic construction site showing a foreman with a weathered face in a yellow hardhat pointing at blueprints, while two workers carry a steel beam.",
            "nouns": ["foreman", "worker", "steel beam", "blueprint"],
        },
        {
            "prompt": "A photorealistic hospital room where a nurse with short curly red hair gently adjusts a patient's IV drip, while a doctor in a white coat reviews charts.",
            "nouns": ["nurse", "patient", "doctor", "IV drip", "chart"],
        },
        {
            "prompt": "A photorealistic classroom scene featuring a male teacher with rolled-up sleeves writing on a whiteboard, while students in uniforms attentively raise hands, and open textbooks lie on desks.",
            "nouns": ["teacher", "whiteboard", "student", "textbook", "desk"],
        },
        {
            "prompt": "A photorealistic street market where a vibrant vendor wearing a striped apron arranges fresh produce, a customer with a wide-brimmed hat inspects fruits in a shopping basket.",
            "nouns": ["vendor", "produce", "customer", "shopping basket"],
        },
        {
            "prompt": "A photorealistic fire station showing a strong firefighter in full gear testing a hose nozzle near a shiny red fire truck, with tools neatly organized.",
            "nouns": ["firefighter", "hose nozzle", "fire truck", "tool"],
        },
        {
            "prompt": "A photorealistic café where a barista with elaborate tattooed arms steams milk behind the espresso counter, a businessman checks his watch while waiting by the display case.",
            "nouns": ["barista", "espresso counter", "businessman", "display case", "milk"],
        },
        {
            "prompt": "A photorealistic park scene with a focused jogger in neon running shoes stretching near a wooden bench, a smiling mother pushing a stroller, and ducks swimming peacefully in a pond.",
            "nouns": ["jogger", "bench", "mother", "stroller", "duck", "pond"],
        },
        {
            "prompt": "A photorealistic airport security line showing a tired traveler in a hoodie placing luggage on the conveyor belt, TSA agents diligently monitoring screens, and suitcases moving through the scanner.",
            "nouns": ["traveler", "luggage", "conveyor belt", "TSA agent", "scanner"],
        },
        {
            "prompt": "A photorealistic science lab where a meticulous researcher with safety goggles pipettes liquid into test tubes, while a colleague records data on a clipboard near a microscope.",
            "nouns": ["researcher", "test tube", "colleague", "microscope", "clipboard"],
        },
        {
            "prompt": "A photorealistic courtroom with a stoic judge in black robes listening intently to a lawyer pointing at evidence, and jurors taking notes in the jury box.",
            "nouns": ["judge", "lawyer", "evidence", "juror", "jury box"],
        },
    ],
    "anime": [
        {
            "prompt": "An anime-style classroom where a student with spiky blue hair sleeps soundly at his desk, a strict teacher with round glasses writes on the chalkboard, and classmates pass notes.",
            "nouns": ["student", "desk", "teacher", "chalkboard", "classmate"],
        },
        {
            "prompt": "An anime-style battle scene featuring a heroic swordsman with a red headband deflecting arrows with his katana, a powerful mage in flowing robes chanting spells, and flying monsters.",
            "nouns": ["swordsman", "arrow", "katana", "mage", "flying monster"],
        },
        {
            "prompt": "An anime-style café where a cheerful waitress with twin-tails carries a tray of drinks, a cat-eared barista steams milk behind the counter, and customers sit at tables.",
            "nouns": ["waitress", "tray", "barista", "customer", "table"],
        },
        {
            "prompt": "An anime-style spaceship bridge showing a focused pilot with cybernetic implants adjusting holographic controls, an engineer repairing console wiring, and a robot co-pilot.",
            "nouns": ["pilot", "holographic control", "engineer", "console", "robot co-pilot"],
        },
        {
            "prompt": "An anime-style festival scene with a serene shrine maiden in traditional dress ringing a large bell, festival-goers carrying paper lanterns, and food stalls serving takoyaki.",
            "nouns": ["shrine maiden", "bell", "festival-goer", "paper lantern", "food stall", "takoyaki"],
        },
    ],
    "watercolor": [
        {
            "prompt": "A watercolor painting of an old fisherman with a deeply weathered face mending fishing nets on a wooden dock, while seagulls circle above lobster traps.",
            "nouns": ["fisherman", "fishing net", "dock", "seagull", "lobster trap"],
        },
        {
            "prompt": "A watercolor scene showing an artist with paint-splattered jeans working at an easel in a vibrant sunflower field, her straw hat hanging from the easel.",
            "nouns": ["artist", "easel", "sunflower", "straw hat"],
        },
        {
            "prompt": "A watercolor market stall where a friendly vendor in a striped apron arranges melons, a grandmother with a woven basket selects fresh vegetables, and baskets of colorful spices emit fragrant powders.",
            "nouns": ["vendor", "melon", "grandmother", "woven basket", "vegetable", "spice"],
        },
    ],
    "cyberpunk": [
        {
            "prompt": "A neon-lit cyberpunk alley where a mysterious hacker with a glowing eye implant types on a holographic keyboard, a towering cyborg bouncer scans IDs at a club entrance, and drones patrol.",
            "nouns": [
                "hacker",
                "holographic keyboard",
                "cyborg bouncer",
                "club entrance",
                "drone",
            ],
        },
        {
            "prompt": "A cyberpunk police station showing a gritty detective with a neural interface reviewing case files on transparent screens, while an arrested cyborg is restrained in an interrogation chair.",
            "nouns": ["detective", "case file", "transparent screen", "cyborg", "interrogation chair"],
        },
        {
            "prompt": "A cyberpunk noodle stall where a skilled chef with robotic arms prepares ramen, a weary salaryman checks his wrist display, and steam rises from boiling pots.",
            "nouns": ["chef", "ramen", "salaryman", "boiling pot"],
        },
    ],
}


EXAMPLE_FORMATS_2 = {
    "realistic": [
        {
            "prompt": "A photorealistic construction site showing a foreman with a weathered face in a yellow hardhat and reflective vest pointing at blueprints, while two workers carry steel beams in the background to the right, and a large crane lifts materials overhead on the left, with excavators moving dirt below.",  # Modified example
            "nouns": ["foreman", "worker", "crane", "excavator", "steel beam", "blueprint"],
        },
        {
            "prompt": "A photorealistic hospital room where a nurse with short curly red hair gently adjusts a patient's IV drip on the left side of the bed, while a doctor in a white coat reviews charts at the foot of the bed on the right, and various medical equipment stands surround the patient in the bed.",
            "nouns": ["nurse", "patient", "doctor", "medical equipment stand", "chart", "IV drip"],
        },
        {
            "prompt": "A photorealistic classroom scene featuring a male teacher with rolled-up sleeves and a friendly expression writing equations on a whiteboard at the front, while students in uniforms attentively raise hands in the middle rows, and open textbooks lie on wooden desks, alongside backpacks on chairs behind them.",
            "nouns": ["teacher", "whiteboard", "student", "textbook", "desk", "backpack"],
        },
        {
            "prompt": "A photorealistic street market where a vibrant vendor wearing a striped apron arranges fresh produce on a stall to the left, a customer with a wide-brimmed hat inspects fruits in a shopping basket in the foreground, and delivery workers carry crates to a produce truck parked nearby in the background.",
            "nouns": ["vendor", "produce", "customer", "shopping basket", "delivery worker", "crate", "produce truck"],
        },
        {
            "prompt": "A photorealistic fire station showing a strong firefighter in full gear testing a hose nozzle on the right, another firefighter checking oxygen tanks near a shiny red fire truck in the center, with tools neatly organized in the truck's open compartments.",
            "nouns": ["firefighter", "hose nozzle", "oxygen tank", "fire truck", "tool"],
        },
        {
            "prompt": "A photorealistic café where a barista with elaborate tattooed arms steams milk behind the espresso counter on the left, a businessman checks his watch while waiting by the display case in the middle, and fresh pastries are displayed under glass domes, with coffee machines whirring behind them.",
            "nouns": ["barista", "espresso counter", "businessman", "display case", "pastry", "coffee machine"],
        },
        {
            "prompt": "A photorealistic park scene with a focused jogger in neon running shoes stretching near a wooden bench on the left, a smiling mother pushing a stroller on the right, ducks swimming peacefully in a pond in the foreground, and elegant willow trees gracefully swaying in the background.",
            "nouns": ["jogger", "bench", "mother", "stroller", "duck", "pond", "willow tree"],
        },
        {
            "prompt": "A photorealistic airport security line showing a tired traveler in a hoodie placing luggage on the conveyor belt to the left, TSA agents diligently monitoring screens above, and suitcases moving through the X-ray scanner, with metal detectors standing by on the right.",
            "nouns": [
                "traveler",
                "luggage",
                "conveyor belt",
                "TSA agent",
                "X-ray scanner",
                "metal detector",
                "security screen",
            ],
        },
        {
            "prompt": "A photorealistic science lab where a meticulous researcher with safety goggles pipettes liquid into test tubes on the workbench, while a colleague records data on a clipboard near a microscope to the right, and beakers bubble on a hot plate in front.",
            "nouns": ["researcher", "test tube", "colleague", "microscope", "beaker", "hot plate"],
        },
        {
            "prompt": "A photorealistic courtroom with a stoic judge in black robes listening intently to a lawyer pointing at evidence on a projection screen to the left, jurors taking notes in the jury box on the right, and a bailiff standing by the witness box in front, with courtroom benches filling the space behind.",
            "nouns": [
                "judge",
                "lawyer",
                "evidence",
                "juror",
                "bailiff",
                "witness box",
                "courtroom bench",
                "projection screen",
            ],
        },
    ],
    "anime": [
        {
            "prompt": "An anime-style classroom where a student with spiky blue hair sleeps soundly at his desk in the front row, a strict teacher with round glasses writes on the chalkboard above, classmates pass notes between rows of wooden desks, and school bags are piled in a corner behind them.",
            "nouns": ["student", "desk", "teacher", "chalkboard", "classmate", "school bag"],
        },
        {
            "prompt": "An anime-style battle scene featuring a heroic swordsman with a red headband deflecting arrows with his katana in the center, a powerful mage in flowing robes chanting spells to the right, magical runes glowing on ancient stone pillars above, and flying monsters in the distance.",
            "nouns": ["swordsman", "arrow", "katana", "mage", "magic rune", "stone pillar", "flying monster"],
        },
        {
            "prompt": "An anime-style café where a cheerful waitress with twin-tails carries a tray of drinks to the left, a cat-eared barista steams milk behind the counter on the right, customers sit at marble-top tables under hanging lanterns, and plush toys decorate shelves above.",
            "nouns": ["waitress", "tray", "barista", "customer", "marble-top table", "hanging lantern", "plush toy"],
        },
        {
            "prompt": "An anime-style spaceship bridge showing a focused pilot with cybernetic implants adjusting holographic controls in the foreground, an engineer repairing console wiring to the left, alien artifacts displayed in crystal cases behind, and a robot co-pilot at another station on the right.",
            "nouns": [
                "pilot",
                "holographic control",
                "engineer",
                "console",
                "alien artifact",
                "crystal case",
                "robot co-pilot",
            ],
        },
        {
            "prompt": "An anime-style festival scene with a serene shrine maiden in traditional dress ringing a large bell in the center, festival-goers carrying paper lanterns to the left, food stalls serving takoyaki on wooden skewers to the right, and colorful banners decorating the streets above.",
            "nouns": [
                "shrine maiden",
                "bell",
                "festival-goer",
                "paper lantern",
                "food stall",
                "takoyaki",
                "colorful banner",
            ],
        },
    ],
    "watercolor": [
        {
            "prompt": "A watercolor painting of an old fisherman with a deeply weathered face mending fishing nets on a wooden dock to the left, while seagulls circle above lobster traps on the right, and sailboats gently bob in the harbor behind, with buoys floating nearby in the foreground.",
            "nouns": ["fisherman", "fishing net", "dock", "seagull", "lobster trap", "sailboat", "harbor", "buoy"],
        },
        {
            "prompt": "A watercolor scene showing an artist with paint-splattered jeans working at an easel in a vibrant sunflower field in the center, her straw hat hanging from the easel, paintbrushes soaking in a mason jar to the left, and a picnic blanket laid out nearby to the right.",
            "nouns": ["artist", "easel", "sunflower", "paintbrush", "mason jar", "picnic blanket"],
        },
        {
            "prompt": "A watercolor market stall where a friendly vendor in a striped apron arranges melons in front, a grandmother with a woven basket selects fresh vegetables to the right, and baskets of colorful spices emit fragrant powders to the left, with wooden crates stacked beneath.",
            "nouns": ["vendor", "melon", "grandmother", "woven basket", "vegetable", "spice", "wooden crate"],
        },
    ],
    "cyberpunk": [
        {
            "prompt": "A neon-lit cyberpunk alley where a mysterious hacker with a glowing eye implant types on a holographic keyboard to the left, a towering cyborg bouncer scans IDs at a club entrance to the right, drones patrol between pulsating neon signs above, and sleek hovercars zoom overhead.",
            "nouns": [
                "hacker",
                "holographic keyboard",
                "cyborg bouncer",
                "club entrance",
                "drone",
                "neon sign",
                "hovercar",
            ],
        },
        {
            "prompt": "A cyberpunk police station showing a gritty detective with a neural interface reviewing case files on transparent screens in the center, while an arrested cyborg with powerful mechanical arms is restrained in an interrogation chair to the right, and data pads lie on the table in front.",
            "nouns": ["detective", "case file", "transparent screen", "cyborg", "interrogation chair", "data pad"],
        },
        {
            "prompt": "A cyberpunk noodle stall where a skilled chef with robotic arms prepares ramen behind the counter, a weary salaryman checks his wrist display to the left, steam rises from boiling pots under holographic menu boards above, and empty bowls are stacked high to the right.",
            "nouns": [
                "chef",
                "ramen",
                "salaryman",
                "boiling pot",
                "holographic menu board",
                "empty bowl",
                "noodle stall",
            ],
        },
    ],
}


def generate_initial_prompt(num_prompts=30):
    style_name = random.choices(list(STYLE_TEMPLATES.keys()), weights=[0.75, 0.15, 0.05, 0.05], k=1)[0]
    style = STYLE_TEMPLATES[style_name]
    random_example = random.choice(EXAMPLE_FORMATS[style_name])
    # avoid using single quotes in JSON when dict is converted to string
    random_example_nouns_str = json.dumps(random_example["nouns"])

    return dedent(f"""\
    Generate {num_prompts} scene descriptions in JSON format with:
    - Style: You should add the prefix "{style["prefix"]}" to each prompt.
    - Key rules:
      * {GENERATION_RULES["instance"]["count"]}
      * {GENERATION_RULES["instance"]["size"]}
      * {GENERATION_RULES["instance"]["id"]}
      * Background: {GENERATION_RULES["background"]["description"]}
      * Constraint: {GENERATION_RULES["instance"]["constraint"]}
    - Directly give the JSON format without any additional text or explanation.
    
    Example structure:
    {{"prompt": "{random_example["prompt"]}", "nouns": {random_example_nouns_str}}}
    """).strip()


def generate_initial_prompt_2(num_prompts_per_call=30):
    style_name = random.choices(list(STYLE_TEMPLATES.keys()), weights=[0.75, 0.15, 0.05, 0.05], k=1)[0]
    style = STYLE_TEMPLATES[style_name]
    random_example = random.choice(EXAMPLE_FORMATS_2[style_name])
    random_example_nouns_str = json.dumps(random_example["nouns"])

    return dedent(f"""\
    Generate {num_prompts_per_call} scene descriptions in JSON format with:
    - Style: You should add the prefix "{style["prefix"]}" to each prompt.
    - Key rules:
      * {GENERATION_RULES_2["instance"]["count"]}
      * {GENERATION_RULES_2["instance"]["size"]}
      * {GENERATION_RULES_2["instance"]["id"]}
      * Background: {GENERATION_RULES_2["background"]["description"]}
      * Spatial cues: {GENERATION_RULES_2["background"]["spatial_cues"]}
      * Constraint: {GENERATION_RULES_2["instance"]["constraint"]}
    - Directly give the JSON format without any additional text or explanation.
    
    Example structure:
    {{"prompt": "{random_example["prompt"]}", "nouns": {random_example_nouns_str}}}
    """).strip()


def llm_prompt_generator(save_path, num_prompts_per_call=30, process_id=0, max_prompts_num=1000):
    balance = check_balance(DEEPSEEK_API_KEY)
    if balance is None:
        print(f"Process {process_id}: Failed to retrieve API balance, stopping generation.")
        return
    else:
        print(f"Process {process_id}: Current API balance: {balance} Yuan")
    batch_count = 0
    valid_prompts_num = 0
    while valid_prompts_num < max_prompts_num:
        initial_prompt = generate_initial_prompt_2(num_prompts_per_call=num_prompts_per_call)
        prompts = generate_prompts(CLIENT, initial_prompt)
        json_prompts = parse_json(prompts)

        if not json_prompts:
            print(f"Process {process_id}, batch {batch_count}: No valid prompts generated.")
        else:
            dump_formatted_json(json_prompts, save_path, indent=1)
            print(
                f"Process {process_id}, batch {batch_count}: Generated {len(json_prompts)} prompts and saved to {save_path}"
            )
            valid_prompts_num += len(json_prompts)

        batch_count += 1

        balance = check_balance(DEEPSEEK_API_KEY)
        if balance is not None and float(balance) < 5:
            print(f"Process {process_id}: API balance is low ({balance} Yuan), stopping generation.")
            break
        else:
            print(f"Process {process_id}: Current API balance: {balance} Yuan, continuing generation.")


def parallel_llm_prompt_generator(save_path_list, num_prompts_per_call=30, num_processes=4, total_prompts_num=50000):
    from multiprocessing import Pool

    with Pool(processes=num_processes) as pool:
        pool.starmap(
            llm_prompt_generator,
            [
                (save_path, num_prompts_per_call, i, total_prompts_num // num_processes)
                for i, save_path in enumerate(save_path_list)
            ],
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate scene description prompts using LLM in parallel.")
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes to use.")
    parser.add_argument(
        "--num_prompts_per_call", type=int, default=30, help="Number of prompts to generate per API call."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DATASET_DIR,
        help="Directory to save generated prompts.",
    )
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    save_path_list = []
    cur_prompt_len = 0
    for i in range(args.num_processes):
        save_path_list.append(os.path.join(args.save_dir, f"prompts_{i + 1}.json"))
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
        total_prompts_num=args.total_prompts - cur_prompt_len if args.total_prompts > cur_prompt_len else 0,
    )

    # gather prompts, assign indices, and save to a single file
    print("Gathering prompts from all processes...")
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


if __name__ == "__main__":
    main()
