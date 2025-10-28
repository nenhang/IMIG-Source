import json
import os
import random
import sys
from pathlib import Path
from textwrap import dedent

from openai import OpenAI
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environ_config import DATASET_DIR_2, DEEPSEEK_API_KEY
from src.utils.api_request import check_balance, generate_prompts
from src.utils.json_formatter import dump_formatted_json, parse_json

DEEPSEEK_API_URL = "https://api.deepseek.com"
CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)

EXAMPLE_FORMATS = {
    # --- 2 Instances (Mostly Human/Animal) ---
    "two_instances": [
        # 人-物
        """
        "prompt": "A young man is riding a bright red bicycle down a city street, with his feet on the pedals.",
        "instance": ["man", "bicycle"],
        "instance_prompt": ["A young man with short brown hair and a clean-shaven face", "a bright red vintage road bicycle with chrome handlebars"]
        """,
        # 人-物
        """
        "prompt": "A woman is wearing a colorful scarf wrapped loosely around her neck on a chilly day.",
        "instance": ["woman", "scarf"],
        "instance_prompt": ["A woman with long black hair and a calm expression", "a large, multicolored wool knit scarf"]
        """,
        # 人-物
        """
        "prompt": "A chef is slicing a ripe tomato on a wooden cutting board with a sharp knife.",
        "instance": ["chef", "tomato"],
        "instance_prompt": ["A male chef with a focused look and a white uniform", "a single, perfectly ripe red heirloom tomato"]
        """,
        # 人-物
        """
        "prompt": "A hiker is holding a walking stick, using it to navigate a rocky path.",
        "instance": ["hiker", "walking stick"],
        "instance_prompt": ["A middle-aged hiker with a weathered face and a slight beard", "a tall, sturdy wooden walking stick with a leather strap"]
        """,
        # 动物-物
        """
        "prompt": "A fluffy dog is chewing on a large, red toy ball, holding it with its paws.",
        "instance": ["dog", "toy ball"],
        "instance_prompt": ["A fluffy golden retriever dog with a wagging tail", "a large, bright red rubber toy ball"]
        """,
        # 动物-物
        """
        "prompt": "A small bird is perched on the rim of an empty birdbath.",
        "instance": ["bird", "birdbath"],
        "instance_prompt": ["A small bright yellow canary bird with black markings", "an empty stone birdbath with a weathered patina"]
        """,
    ],
    # --- 3 Instances (High Human/Animal Focus, Object-Object removed) ---
    "three_instances": [
        # 人-人-物 (Multi-Human)
        """
        "prompt": "A photographer is sharing a single pair of binoculars with a tourist, with the binoculars resting on their shoulder.",
        "instance": ["photographer", "tourist", "binoculars"],
        "instance_prompt": ["A man photographer with short hair and a determined expression", "A woman tourist with long curly hair and a curious gaze", "a single black pair of binoculars"]
        """,
        # 人-人-物 (Multi-Human)
        """
        "prompt": "A child is handing a flower to their mother, who is holding a wicker basket that is resting against her hip.",
        "instance": ["child", "mother", "flower"],
        "instance_prompt": ["A smiling young child with rosy cheeks and blonde hair", "A woman with short blonde hair and a gentle smile", "a bright red rose flower"]
        """,
        # 人-物-物
        """
        "prompt": "An artist is holding a palette covered in paint, using a brush to touch up a portrait on an easel.",
        "instance": ["artist", "palette", "brush"],
        "instance_prompt": ["A painter with an intense focus and neatly combed hair", "a large wooden palette covered in oil paint", "a fine-tipped sable hair brush"]
        """,
        # 人-动物-物
        """
        "prompt": "A girl is petting a large white cat that is sitting on a cozy blue blanket.",
        "instance": ["girl", "cat", "blanket"],
        "instance_prompt": ["A happy young girl with braided hair and a joyful expression", "a large white Persian cat", "a cozy dark blue knitted blanket"]
        """,
        # 动物-物-物
        """
        "prompt": "A sleek black cat is sleeping soundly on top of a comfortable, gray pillow, with a remote control partially covered by its paw.",
        "instance": ["cat", "pillow", "remote control"],
        "instance_prompt": ["A sleek short-haired black cat", "a comfortable, fluffy gray square pillow", "a black television remote control"]
        """,
        # 人-物-物
        """
        "prompt": "A waiter is placing a ceramic coffee cup onto a wooden saucer, which is resting on a tablecloth.",
        "instance": ["waiter", "coffee cup", "spoon"],
        "instance_prompt": ["A formal waiter with slicked-back hair and a neutral expression", "a delicate white ceramic coffee cup", "a shiny silver teaspoon"]
        """,
    ],
    "four_instances": [
        # 人-人-物-物 (Multi-Human)
        """
        "prompt": "A teacher is handing a red apple to a student, who is sitting at a wooden desk with an open textbook resting on their lap.",
        "instance": ["teacher", "student", "apple", "textbook"],
        "instance_prompt": ["An adult female teacher with shoulder-length hair and a professional expression", "A young student with short dark hair and a curious gaze", "a single bright red Granny Smith apple", "an open science textbook with colorful diagrams"]
        """,
        # 人-人-物-物 (Multi-Human)
        """
        "prompt": "A mechanic is inspecting an engine, while an assistant holds a flashlight pressed against the engine block, with a wrench gripped in the mechanic's hand.",
        "instance": ["mechanic", "assistant", "engine", "wrench"],
        "instance_prompt": ["A mechanic with a focused brow and oily stubble", "An assistant with a neat haircut and a careful expression", "the rusty V8 car engine block", "a large chrome adjustable wrench"]
        """,
        # 人-动物-动物-物 (NEW: 1 Human + 2 Animal Example)
        """
        "prompt": "A boy is holding a small brown lizard, which is resting on his arm, while a small turtle is climbing onto a wet mossy rock.",
        "instance": ["boy", "lizard", "turtle", "rock"],
        "instance_prompt": ["A young boy with messy blonde hair and a happy face", "a small brown speckled lizard", "a small green box turtle", "a large wet mossy gray rock"]
        """,
        # 人-动物-物-物
        """
        "prompt": "A woman is holding a leash attached to a golden retriever that is resting its head on a picnic blanket, with her hand touching the dog's collar.",
        "instance": ["woman", "dog", "leash", "blanket"],
        "instance_prompt": ["A woman with long wavy hair and a relaxed expression", "a large golden retriever dog with soft fur", "a long braided blue nylon leash", "a red and white checkered picnic blanket"]
        """,
        # 动物-物-物-物 (保持不变，因为无人)
        """
        "prompt": "A brown squirrel is burying an acorn in a pile of fallen autumn leaves, with its tail curled against the base of an oak tree.",
        "instance": ["squirrel", "acorn", "leaves", "oak tree"],
        "instance_prompt": ["A fluffy brown bushy-tailed squirrel", "a large brown acorn with a cap", "a pile of fallen red and yellow autumn leaves", "an ancient oak tree with thick bark"]
        """,
        # 人-动物-物-物
        """
        "prompt": "A young boy is gently placing a tennis ball inside a small, empty bird cage, with a fluffy white poodle standing beside him.",
        "instance": ["boy", "poodle", "tennis ball", "bird cage"],
        "instance_prompt": ["A young boy with blonde hair and a gentle expression", "a fluffy white miniature poodle with a collar", "a bright yellow worn tennis ball", "a small empty metal bird cage with a latch"]
        """,
    ],
    # --- 5 Instances (Highest Human/Animal Focus, 2H+2A added, Pure Animal-Object removed) ---
    "five_instances": [
        # 人-人-人-物-物 (3-Human Example)
        """
        "prompt": "Three musicians are on a small stage: one is playing a banjo, another is singing into a microphone, and the third is resting a fiddle on their shoulder.",
        "instance": ["banjo player", "singer", "fiddler", "banjo", "microphone"],
        "instance_prompt": ["A man with a full beard and a passionate expression", "A woman vocalist with long curly hair and dramatic makeup", "A third person with spectacles and a focused expression", "a five-string wooden banjo", "a tall chrome stage microphone"]
        """,
        # 人-动物-动物-物-物 (1 Human + 2 Animal Example #1)
        """
        "prompt": "A child is gently petting a fluffy white rabbit that is sitting on a patch of grass, while a small brown mouse runs across a nearby wooden fence.",
        "instance": ["child", "rabbit", "mouse", "grass", "fence"],
        "instance_prompt": ["A happy young child with sun-kissed hair and bright eyes", "a fluffy white Angora rabbit with pink eyes", "a small brown field mouse", "a patch of bright green short grass", "a sturdy weathered wooden fence"]
        """,
        # 人-动物-动物-物-物 (NEW 1 Human + 2 Animal Example #2)
        """
        "prompt": "A man is holding a fishing net, which contains a struggling silver fish and a small crab, while standing on a rocky shoreline.",
        "instance": ["man", "fish", "crab", "fishing net", "shoreline"],
        "instance_prompt": ["A fisherman with a heavy beard and intense gaze", "a large struggling silver salmon fish", "a small red Dungeness crab", "a large green nylon fishing net", "a grey rocky shoreline covered in barnacles"]
        """,
        # 人-人-物-物-物 (Multi-Human)
        """
        "prompt": "A mother is helping her child tie a red shoelace on a canvas sneaker, sitting on a wooden bench, with her hands gripping the shoe.",
        "instance": ["mother", "child", "shoelace", "sneaker", "bench"],
        "instance_prompt": ["A smiling adult mother with long dark hair", "A young child with a curious face and blonde hair", "a thin red fabric shoelace", "a white canvas high-top sneaker", "a worn wooden park bench with chipped paint"]
        """,
        # 人-人-物-物-物 (Multi-Human)
        """
        "prompt": "A musician is playing a guitar, while an audience member records the performance with a smartphone, with the microphone positioned in front of the guitar's sound hole.",
        "instance": ["musician", "audience member", "guitar", "smartphone", "microphone"],
        "instance_prompt": ["A musician with short spiky hair and sharp features", "An audience member with a short haircut and interested expression", "a glossy black acoustic guitar", "a silver metallic smartphone", "a professional black stage microphone on a stand"]
        """,
        # 人-动物-物-物-物
        """
        "prompt": "A hiker is feeding a small deer an apple, with the apple pressed into the deer's mouth, while the hiker's hand rests on a green hiking backpack which is against a sturdy wooden fence.",
        "instance": ["hiker", "deer", "apple", "backpack", "fence"],
        "instance_prompt": ["A hiker with medium-length hair and a calm expression", "a small brown doe deer", "a crisp red gala apple", "a large green hiking backpack with orange straps", "a sturdy wooden split-rail fence"]
        """,
    ],
}


# --- PROBABILITY MAPPING (Unchanged) ---
INSTANCE_COUNT_PROBS = {
    2: 1,
    3: 3,
    4: 3,
    5: 1,
}

COUNT_TO_CATEGORY_MAP = {
    2: ["two_instances"],
    3: ["three_instances"],
    4: ["four_instances"],
    5: ["five_instances"],
}


# --- generate_initial_prompt FUNCTION ---
def generate_initial_prompt(num_prompts=40):
    """
    Generates the initial prompt for the LLM, with examples corresponding to a
    randomly selected instance count based on defined probabilities,
    and includes explicit instructions tailored to the current instance count.
    """

    # 1. Select the number of instances based on the custom probabilities
    instance_counts = list(INSTANCE_COUNT_PROBS.keys())
    weights = list(INSTANCE_COUNT_PROBS.values())

    selected_count = random.choices(instance_counts, weights=weights, k=1)[0]

    # 2. Select two unique example templates for the chosen instance count
    category_key = COUNT_TO_CATEGORY_MAP[selected_count][0]
    examples = EXAMPLE_FORMATS[category_key]

    example_1_str, example_2_str = random.sample(examples, 2)
    example_1 = example_1_str.strip()
    example_2 = example_2_str.strip()

    # 3. Dynamic Construction of the Human Prioritization Instruction
    if selected_count <= 3:
        human_priority_instruction = "Aim for at least one human subject in most of the generated prompts."
        num_prompts_ = min(num_prompts, 40)  # Increase number of prompts for 2 or 3 instances
    else:
        # Applies to 4, and 5 instances
        human_priority_instruction = (
            "Aim to include **two or more distinct human subjects (e.g., 'man' and 'woman', or 'teacher' and 'student')** "
            "in the majority (at least 75%) of the generated prompts."
        )
        num_prompts_ = min(num_prompts, 30)  # Decrease number of prompts for 4 or 5 instances to ensure quality

    # 4. Construct the prompt with the dynamic instruction
    initial_prompt = f"""
Basic requirements:
Generate {num_prompts_} scene descriptions in JSON format. Each entry must describe a scene with **exactly {selected_count} common subjects** with a simple, clear, and direct interaction that results in a visible physical connection or overlap in an image. The prompts should be natural and realistic, using common verbs and prepositions that imply this connection.

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
The "instance" list must contain exactly **{selected_count}** specific, singular nouns or noun phrases, each representing a distinct subject for target detection. **IMPORTANT:** All items in the "instance" list MUST be unique and clean base nouns (e.g., 'teacher', 'student', 'apple'). **AVOID NUMBERED INSTANCES**: Do not use generic numbered names like 'musician 1', 'person 2', or 'cat 3'. Instead, use a unique and descriptive noun or noun phrase for each distinct subject (e.g., 'banjo player', 'singer', 'fiddler'). **CRITICAL INSTANCE INDEPENDENCE**: Ensure the instances selected for the "instance" list are visually and conceptually distinct. Avoid selecting sets of highly-dependent objects that are typically seen as a single functional unit (e.g., a camera mounted on a tripod with a flash).
The "instance_prompt" list must contain **{selected_count}** independent phrases. Each phrase must be a stand-alone description of the corresponding subject in the "instance" list, focusing on its appearance and attributes. **STRICTLY FORBIDDEN:** In "instance_prompt", do not use verbs or phrases that describe the subject's *action* or *relationship to the main prompt* (e.g., 'a man riding a bike', 'a customer waiting', 'somebody doing something'). The descriptions must be entirely **STATIC**. Do not use any relative or ordinal descriptive words such as 'another', 'second', even for multi-human subjects. The instance_prompt description must be detailed enough to allow for the generation of a clear, unambiguous instance. Avoid overly generic nouns (e.g., "a net") and instead use more specific descriptions (e.g., "a basketball net").
The "prompts" should use a mix of dynamic verbs like 'riding', 'sitting', 'carrying' or clear positional prepositions like 'inside of', 'against' to ensure visual overlap. Crucially, the prompt must describe a clear and sufficient visual interaction or physical connection (touching, holding, resting on, etc.) between all {selected_count} instances.
**STRONG PRIORITIZATION FOR HUMAN SUBJECTS:** The generated prompts MUST have a high percentage of human subjects.
* {human_priority_instruction}
**PERSON DESCRIPTION PRIORITY:** For any human subject, the "instance_prompt" must primarily focus on **facial features and expression**. Clothing and accessory (glasses, hats, jewelry, etc.) descriptions are secondary and should only be included in **a few** cases to add variety.
Ensure the generated prompts and subject pairs are highly diverse and do not repeat or closely resemble the provided examples.
Avoid vague terms (sky, ground, water, etc.) and scene-related terms (forest, street, etc.).
Output ONLY the JSON array, nothing else. Do not include any additional text or explanations. Now generate {num_prompts_} scene descriptions in the specified JSON format.
"""
    return dedent(initial_prompt).strip()


def llm_prompt_generator(save_path, num_prompts_per_call=40, process_id=0, balance_threshold=15, num_prompts_need=None):
    """
    Main function to generate prompts in a loop until balance is low.
    """
    if num_prompts_need is not None and num_prompts_need <= 0:
        print(f"Process {process_id}: No prompts needed, exiting.")
        return

    save_path = save_path[0] if isinstance(save_path, list) else save_path
    balance = check_balance(DEEPSEEK_API_KEY)
    if balance is None:
        print(f"Process {process_id}: Failed to retrieve API balance, stopping generation.")
        return
    else:
        print(f"Process {process_id}: Current API balance: {balance} Yuan")

    batch_count = 0
    num_prompts_generated = 0
    with tqdm(total=num_prompts_need, desc=f"Process {process_id} Prompt Generation") as pbar:
        while True:
            initial_prompt = generate_initial_prompt(num_prompts=num_prompts_per_call)
            prompts = generate_prompts(CLIENT, initial_prompt, temperature=1.3)
            json_prompts = parse_json(prompts)

            if not json_prompts:
                print(f"Process {process_id}, batch {batch_count}: No valid prompts generated.")
            else:
                valid_prompts = []
                for p in json_prompts:
                    if (
                        p.get("instance")
                        and 2 <= len(p["instance"]) <= 5
                        and p.get("instance_prompt")
                        and len(p["instance_prompt"]) == len(p["instance"])
                    ):
                        valid_prompts.append(p)

                if not valid_prompts:
                    print(
                        f"Process {process_id}, batch {batch_count}: Generated prompts but none had the correct format."
                    )
                else:
                    dump_formatted_json(valid_prompts, save_path)
                    print(
                        f"Process {process_id}, batch {batch_count}: Generated {len(valid_prompts)} prompts and saved to {save_path}"
                    )
                    num_prompts_generated += len(valid_prompts)
                    pbar.update(len(valid_prompts))
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
    parser.add_argument("--total_prompts", type=int, default=50000, help="Total number of prompts to generate.")
    parser.add_argument("--num_processes", type=int, default=64, help="Number of parallel processes.")
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
    final_save_path = os.path.join(args.save_dir, "prompts.json")
    cur_prompt_len = 0
    for i in range(args.num_processes):
        save_path_list.append(os.path.join(args.save_dir, f"multi_composite_temp_{i + 1}.json"))
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
    begin_idx = 100000
    for i, prompt in enumerate(all_prompts):
        prompt["index"] = begin_idx + i

    dump_formatted_json(all_prompts, final_save_path)
    print(f"Total {len(all_prompts)} prompts saved to {final_save_path}")
