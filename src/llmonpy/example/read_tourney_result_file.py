from trace_log import TourneyResult

if __name__ == "__main__":
    result_list = TourneyResult.from_file("docs/sample_training_data.json")
    print("Results:")
    for result in result_list:
        print(result.to_json())