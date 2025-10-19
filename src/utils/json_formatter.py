import json
import os


def is_leaf(element):
    """判断元素是否为叶子节点（基础类型）"""
    return isinstance(element, (str, int, float, bool)) or element is None


def custom_format(obj, indent=0):
    """改进版格式化函数，精确控制缩进"""
    buffer = []
    if isinstance(obj, dict):
        # 字典起始缩进
        buffer.append(f"{'  ' * indent}{{\n")
        items = obj.items()
        for i, (k, v) in enumerate(items):
            # 键的缩进比字典多一级
            buffer.append(f"{'  ' * (indent + 1)}{json.dumps(k)}: ")
            buffer.append(custom_format(v, indent + 1))
            buffer.append(",\n" if i < len(items) - 1 else "\n")
        # 字典结束对齐
        buffer.append(f"{'  ' * indent}}}")
    elif isinstance(obj, list):
        if all(is_leaf(e) for e in obj):
            buffer.append(json.dumps(obj))
        else:
            # 复杂列表起始缩进
            buffer.append(f"[\n{'  ' * (indent + 1)}")
            for i, e in enumerate(obj):
                buffer.append(custom_format(e, indent + 1))
                if i < len(obj) - 1:
                    buffer.append(f",\n{'  ' * (indent + 1)}")
            buffer.append(f"\n{'  ' * indent}]")
    else:
        buffer.append(json.dumps(obj))
    return "".join(buffer)


def format_data(data, indent=1):
    """格式化数据为字符串"""
    if isinstance(data, dict):
        formatted = custom_format(data, indent)
    elif isinstance(data, list):
        entries = [custom_format(item, indent) for item in data]
        formatted = "[\n" + ",\n".join(entries) + "\n]"
    else:
        formatted = json.dumps(data)
    return formatted


def dump_formatted_json(data, target_file, indent=1):
    """将格式化后的数据写入文件"""
    if os.path.exists(target_file):
        with open(target_file, "r+", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    existing_data.extend(data)
                else:
                    raise ValueError("Existing data is not a list, cannot append new items.")
            except json.JSONDecodeError:
                raise ValueError("Existing file is not a valid JSON.")
    else:
        existing_data = data
    formatted_existing = format_data(existing_data, indent)
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(formatted_existing)


def parse_json(data):
    try:
        json_data = json.loads(data)

        if isinstance(json_data, list):
            return json_data
        else:
            print(f"Unexpected JSON format {type(json_data)}, expected a list.")
            return []
    except json.JSONDecodeError:
        start_idx = data.find("[")
        end_idx = data.rfind("]") + 1
        if start_idx != -1 and end_idx != -1:
            json_str = data[start_idx:end_idx]
            try:
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError:
                print("Could not extract valid JSON from response")
                return []
        else:
            print("No valid JSON found in response")
            return []
