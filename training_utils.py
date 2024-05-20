from pathlib import Path
import json
import os


def create_path_and_get_next_idx(outfile, overwrite=False):
    p = Path(outfile)
    # check
    if p.is_dir():
        raise ValueError(f"Output file {outfile} cannot be a directory.")

    # create output dir
    p.parent.mkdir(parents=True, exist_ok=True)

    # overwrite file if asked
    if overwrite:
        with open(outfile, "w") as f:
            return 0

    # get num lines
    if p.exists():
        with open(p.as_posix(), "rb") as f:
            return sum(1 for _ in f)
    else:
        with open(p.as_posix(), "w") as f:
            return 0


def write_record_to_jsonl(outfile, item):
    with open(outfile, "a") as f:
        f.write(json.dumps(item) + "\n")


def load_jsonl(file):
    with open(file, 'r') as f:
        data = f.readlines()
    return [json.loads(record) for record in data]


def extract_text_between_tags(text, start_tag="[ANS]", end_tag="[/ANS]"):
    start_index = text.find(start_tag) + len(start_tag)
    end_index = text.find(end_tag)

    # Extract the text between these indices, stripping any leading/trailing whitespace.
    extracted_text = text[start_index:end_index].strip() if start_index > len(start_tag) - 1 and end_index != -1 else ""

    return extracted_text


def write_jsonl(data, outfile):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, 'w') as f:
        for record in data:
            json_record = json.dumps(record)
            f.write(json_record + '\n')


def unflatten(flattened_list, lengths):
    """
    Reconstructs the original list of lists from a flattened list and the lengths of the original sublists.

    Parameters:
    - flattened_list: A list containing all elements from the original list of lists.
    - lengths: A list of integers where each integer represents the length of a sublist in the original list.

    Returns:
    A list of lists reconstructed based on the provided lengths.
    """
    unflattened = []
    start = 0
    for length in lengths:
        # Extract the sublist using the current start index and the length
        end = start + length
        sublist = flattened_list[start:end]
        unflattened.append(sublist)
        # Update the start index for the next iteration
        start = end
    return unflattened