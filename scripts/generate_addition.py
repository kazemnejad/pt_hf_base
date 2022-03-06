from pathlib import Path

import jsonlines
import numpy as np
from transformers import set_seed


def split_digits(a):
    # introduce space between digits so that they are tokenized separately
    if type(a) == int:
        return " ".join(list(str(a)))
    elif type(a) == str:
        return " ".join(list(a))


def generate_scratchpad(a, b):
    # use the gradeschool carry method
    x = str(a)
    y = str(b)
    scratchpad = ["<scratch>"]
    # scratchpad = ["scratch"]
    scratchpad.append(" ".join([split_digits(a), "+", split_digits(b), ",", "C:", "0"]))

    # pad numbers as necessary
    max_len = max(len(x), len(y))
    x = x.zfill(max_len)
    y = y.zfill(max_len)

    carry = 0
    result = [""] * len(str(a + b))
    for i in range(max_len - 1, -1, -1):
        digit_sum = int(x[i]) + int(y[i]) + carry
        carry = 0 if digit_sum < 10 else 1
        result[i] = str(digit_sum)[-1]
        if i > 0:
            scratchpad.append(
                " ".join(
                    [
                        split_digits(x[:i]),
                        "+",
                        split_digits(y[:i]),
                        ",",
                        *result[i:],
                        "C:",
                        str(carry),
                    ]
                )
            )
        else:
            scratchpad.append(" ".join([",", *result, "C:", str(carry)]))
    scratchpad.append("</scratch>")
    # scratchpad.append("answer")

    input = " ".join(
        [split_digits(a), "+", split_digits(b)]
    )  # make sure we don't put the padding in the input
    final_num = " ".join([str(carry), *result])
    # scratchpad.append(final_num)
    answer = final_num if carry == 1 else final_num[1:]
    # target = [*scratchpad, answer]

    # assert int(final_num.replace(" ", "")) == a + b

    return input, answer, scratchpad


def generate(num_examples, num_digits, strict):
    data = [None] * num_examples

    _example_gen_src = f"{num_digits}_{strict}"

    for i in range(num_examples):
        if strict == "strict":
            # only generate problems with n digits
            a = np.random.randint(
                10 ** (num_digits - 1), 10 ** (num_digits)
            )  # second arg is exclusive with randint
            b = np.random.randint(10 ** (num_digits - 1), 10 ** (num_digits))
        else:
            # ok to generate problems with <= n digits
            a = np.random.randint(
                0, 10 ** (num_digits)
            )  # second arg is exclusive with randint
            b = np.random.randint(0, 10 ** (num_digits))
        source, answer, scratchpad = generate_scratchpad(a, b)
        data[i] = {
            "source": source,
            "target": answer,
            "scratchpad": scratchpad,
            "_gen_src": _example_gen_src,
        }

    return data


if __name__ == "__main__":
    set_seed(12345)

    train_val_idtest_data = generate(110 * 1000, 8, "nostrict")
    train_val, test_iid = train_val_idtest_data[:105000], train_val_idtest_data[105000:]
    train, validation = train_val_idtest_data[:100000], train_val_idtest_data[100000:]

    test_ood = (
        generate(2500, 9, "strict")
        + generate(2500, 10, "strict")
        + generate(2500, 11, "strict")
        + generate(2500, 12, "strict")
    )

    output_path = Path("generated_data/addition/normal")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "train.jsonl", "w") as output:
        jsonlines.Writer(output).write_all(train)

    with open(output_path / "valid.jsonl", "w") as output:
        jsonlines.Writer(output).write_all(validation)

    with open(output_path / "test.iid.jsonl", "w") as output:
        jsonlines.Writer(output).write_all(test_iid)

    with open(output_path / "test.jsonl", "w") as output:
        jsonlines.Writer(output).write_all(test_ood)
