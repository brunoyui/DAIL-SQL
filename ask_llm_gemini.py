import argparse
import os
import json

import google.generativeai as genai
from tqdm import tqdm

from llm.gemini import ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader

QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--gemini_api_key", type=str)
    parser.add_argument("--model", type=str, choices=[LLM.GEMINI_1_0_PRO, 
                                                      LLM.GEMINI_1_0_PRO_001,
                                                      LLM.GEMINI_1_5_PRO],
                        default=LLM.GEMINI_1_0_PRO)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    token_cnt = 0
    with open(out_file, mode) as f:
        for i, batch in enumerate(tqdm(question_loader)):
            if i < args.start_index:
                continue
            if i >= args.end_index:
                break
            try:
                res = ask_llm(args.model, batch, args.gemini_api_key)
            except Exception as e:
                print(f"exception: {e}", end="\n")
                res = ""

            for sql in res.text:
                # remove \n and extra spaces
                sql = " ".join(sql.replace("\n", " ").split())
                # python version should >= 3.8
                if sql.startswith("SELECT"):
                    f.write(sql + "\n")
                elif sql.startswith(" "):
                    f.write("SELECT" + sql + "\n")
                else:
                    f.write("SELECT " + sql + "\n")
                print(sql)
                

