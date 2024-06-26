#   Copyright © 2024 Thomas Edward Burns
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import io
import json
import os

POST_DOUBLE_QUOTE_CHARS = { ":": ":", ",": ",","}": "}" }


def next_nonwhitespace_char(input_string, i):
    result = None
    while i < len(input_string):
        current_char = input_string[i]
        if not current_char.isspace():
            result = current_char
            break
        i += 1
    return result


def last_nonwhitespace_char(input_string, i):
    result = None
    while i >= 0:
        current_char = input_string[i]
        if not current_char.isspace():
            result = current_char
            break
        i -= 1
    return result


# assumes " are balanced if not escaped
def extract_value_string_exclude_last_quote(json_str, start_index):
    in_substring = False
    result = None
    last_char = "\""
    for i in range(start_index, len(json_str)):
        current_char = json_str[i]
        if current_char == "\"" and last_char != "\\":
            if not in_substring and next_nonwhitespace_char(json_str, i+1) in POST_DOUBLE_QUOTE_CHARS:
                result = json_str[start_index:i]
                break
            else:
                in_substring = not in_substring
        last_char = current_char
    return result, i


def handle_json_special_characters_for_values(json_str, start_index, string_builder):
    last_char = None
    value_string, end_of_value_string_index = extract_value_string_exclude_last_quote(json_str, start_index)
    for i in range(len(value_string)):
        current_char = value_string[i]
        if current_char == "\"" and last_char != "\\":
            string_builder.write("\\\"") # escape double quote
        else:
            if current_char == "\n":
                string_builder.write("\\n")
            elif current_char == "\r":
                string_builder.write("\\r")
            elif current_char == "\t":
                string_builder.write("\\t")
            elif current_char == "\b":
                string_builder.write("\\b")
            elif current_char == "\f":
                string_builder.write("\\f")
            elif current_char == "\\":
                if len(value_string) > i+1 and value_string[i+1] != "\\" and value_string[i+1] != "\"" and \
                        value_string[i+1] != "n" and value_string[i+1] != "r" and value_string[i+1] != "t" and \
                        value_string[i+1] != "b" and value_string[i+1] != "f":
                    string_builder.write("\\\\")
                else:
                    string_builder.write(current_char)
            else:
                string_builder.write(current_char)
        last_char = current_char
    string_builder.write("\"")
    return end_of_value_string_index


def fix_common_json_encoding_errors(json_str):
    if json_str is None:
        return None
    result = json_str
    with io.StringIO() as string_builder:
        current_char = None
        last_char = None
        max_index = len(json_str)
        current_char_index = 0
        while current_char_index < max_index:
            current_char = json_str[current_char_index]
            string_builder.write(current_char)
            if current_char == "\"" and last_char != "\\":
                # only handle special characters between quotes for values, not keys.
                if last_nonwhitespace_char(json_str, current_char_index-1) == ":":
                    current_char_index = handle_json_special_characters_for_values(json_str, current_char_index + 1, string_builder)
            last_char = json_str[current_char_index]
            current_char_index += 1
        result = string_builder.getvalue()
    return result


if __name__ == "__main__":
    test_data_directory = "artifacts/json_test_data/"
    working_directory = os.getcwd()
    file_list = os.listdir(test_data_directory)
    for file_name in file_list:
        if file_name.endswith(".json"):
            file_path = os.path.join(test_data_directory, file_name)
            with open(file_path, "r") as file:
                json_str = file.read()
                fixed_json = fix_common_json_encoding_errors(json_str)
                try:
                    json.loads(fixed_json)
                    print(f"JSON is valid. {file_path}")
                except Exception as e:
                    print(f"Error in file {file_path} {str(e)}")
