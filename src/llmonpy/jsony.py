import io
import json

ESCAPE_CHARACTERS = {
    "\b": "b",
    "\f": "f",
    "\n": "n",
    "\r": "r",
    "\t": "t",
    '"': '"'
}


def escape_char(char):
    if char in ESCAPE_CHARACTERS:
        result = '\\' + ESCAPE_CHARACTERS[char]
    elif char < '\x7f':
        result = char
    else:
        result = f'\\u{ord(char):04x}'
    return result


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


def copy_double_quoted_string(jsone_string, i, string_builder):
    string_builder.write(jsone_string[i])
    i += 1
    while i < len(jsone_string):
        current_char = jsone_string[i]
        if current_char == "\"" and jsone_string[i-1] != "\\":
            if next_nonwhitespace_char(jsone_string, i+1) in POST_DOUBLE_QUOTE_CHARS:
                string_builder.write("\"")
                break
            else:
                string_builder.write("\\\"") # escape double quote
        else:
            string_builder.write(current_char)
        i += 1
    return i


def copy_single_quoted_string(jsone_string, i, string_builder):
    string_builder.write("\"")
    i += 1
    while i < len(jsone_string):
        current_char = jsone_string[i]
        if current_char == "'" and jsone_string[i-1] != "\\":
            string_builder.write("\"")
            break
        elif current_char == "\\" and len(jsone_string) > i+1 and jsone_string[i+1] == "'": # escaped single quote, single quote doesn't need to be escaped
            string_builder.write("'")
            i += 1
        else:
            string_builder.write(escape_char(current_char))
        i += 1
    return i


def jsony_to_json(jsone_string):
    if jsone_string is None:
        return None
    result = jsone_string
    with io.StringIO() as string_builder:
        current_char = None
        last_char = None
        max_index = len(jsone_string)
        i = 0
        while i < max_index:
            current_char = jsone_string[i]
            if current_char == "\"" and last_char != "\\":
                i = copy_double_quoted_string(jsone_string, i, string_builder)
            elif current_char == "'":
                i = copy_single_quoted_string(jsone_string, i, string_builder)
            else:
                string_builder.write(current_char)
            last_char = jsone_string[i]  # i could have been modified, so last char if different than current char
            i += 1
        result = string_builder.getvalue()
    return result


# Example usage
def test_jsony_to_json():
    with open("artifacts/jsony_test_data/jsony_input.txt", "r") as file:
        jsone_input = file.read()
        json_output = jsony_to_json(jsone_input)
        result = json.loads(json_output)
        print(result)
        print("jsone done")


if __name__ == "__main__":
    test_jsony_to_json()
    exit(0)