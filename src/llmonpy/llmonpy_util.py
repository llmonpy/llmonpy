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

POST_DOUBLE_QUOTE_CHARS = { ":": ":", ",": ",", "}": "}" }


def next_nonwhitespace_char(jsone_string, i):
    result = None
    while i < len(jsone_string):
        current_char = jsone_string[i]
        if current_char not in [" ", "\t", "\n", "\r"]:
            result = current_char
            break
        i += 1
    return result


def handle_json_special_characters_between_quotes(json_str, start_index, string_builder):
    current_char = None
    last_char = None
    for i in range(start_index, len(json_str)):
        current_char = json_str[i]
        if current_char == "\"" and last_char != "\\":
            if next_nonwhitespace_char(json_str, i+1) in POST_DOUBLE_QUOTE_CHARS:
                string_builder.write("\"")
                break
            else:
                string_builder.write("\\\"") # escape double quote
        else:
            if current_char >= '\x7f':
                string_builder.write(f'\\u{ord(current_char):04x}')
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
                if len(json_str) > i+1 and json_str[i+1] != "\\" and json_str[i+1] != "\"" and \
                        json_str[i+1] != "n" and json_str[i+1] != "r" and json_str[i+1] != "t" and \
                        json_str[i+1] != "b" and json_str[i+1] != "f":
                    string_builder.write("\\\\")
            else:
                string_builder.write(current_char)
        last_char = current_char
    return i


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
                current_char_index = handle_json_special_characters_between_quotes(json_str, current_char_index+1, string_builder)
            last_char = json_str[current_char_index]
            current_char_index += 1
        result = string_builder.getvalue()
    return result