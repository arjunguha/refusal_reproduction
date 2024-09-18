"""

Dataset sources:

- alpaca_data.json:
   https://github.com/tatsu-lab/stanford_alpaca/blob/f13496211289def0ff88ae673389ae14a818b4b3/alpaca_data.json

- MaliciousInstruct.txt:

  https://github.com/Princeton-SysML/Jailbreak_LLM/raw/main/data/MaliciousInstruct.txt


"""
import pandas as pd
from pathlib import Path


def prepare_alpaca():
    df = pd.read_json("datasets/alpaca_data.json")
    return df[["instruction"]]

def prepare_malicious_instruct():
    """
    Read each line into a dataframe with a column "instruction"
    """
    with open("datasets/MaliciousInstruct.txt", "r") as f:
        data = [{"instruction": line.strip()} for line in f]
    df = pd.DataFrame(data, columns=["instruction"])
    return df


def main():
    alpaca_df = prepare_alpaca()
    alpaca_df.to_json(Path("datasets/helpful.jsonl"), orient="records", lines=True)

    malicious_df = prepare_malicious_instruct()
    malicious_df.to_json(Path("datasets/harmful.jsonl"), orient="records", lines=True)


if __name__ == "__main__":
    main()