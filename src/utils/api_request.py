import requests


def check_balance(api_key):
    url = "https://api.deepseek.com/user/balance"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查HTTP错误
        balance_data = response.json()
        balance_info = balance_data.get("balance_infos", {})
        return balance_info[0].get("total_balance", None)
    except requests.exceptions.RequestException as e:
        print(f"Failed to check balance: {e}")
        return None


def generate_prompts(client, initial_prompt, temperature=1.0):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": initial_prompt},
        ],
        stream=False,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
