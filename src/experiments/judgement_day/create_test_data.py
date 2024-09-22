import copy
import json
import os

from experiments.judgement_day.run_test import TestQuestion


def collect_paragraphs(full_test_data):
    paragraphs = []
    data_list = full_test_data["data"]
    for topic in data_list:
        for paragraph in topic["paragraphs"]:
            paragraphs.append(paragraph)
    return paragraphs


def collect_questions(paragraph_list):
    question_list = []
    next_id = 0
    for paragraph in paragraph_list:
        for qa in paragraph["qas"]:
            if qa["is_impossible"] is False:
                question = qa["question"]
                good_answer = qa["answers"][0]["text"]
                context = paragraph["context"]
                question_list.append(TestQuestion(next_id, context, question, good_answer))
                next_id += 1
            else:
                print("Skipping impossible question " + qa["question"])
    return question_list


if __name__ == "__main__":
    api_file_path = os.path.abspath(__file__)
    api_dir = os.path.dirname(api_file_path)
    file_path = api_dir + "/raw_test_data.json"
    with open(file_path, "r") as file:
        full_test_data = json.load(file)
    paragraph_list = collect_paragraphs(full_test_data)
    test_question_list = collect_questions(paragraph_list)
    output_file = api_dir + "/eval_test_data.json"
    test_question_list = test_question_list[:500]
    with open(output_file, "w") as file:
        json.dump([test_question.to_dict() for test_question in test_question_list], file)
    print("done")