# 2024-06-28

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
import os

def bytes_to_bits(data: bytes) -> np.ndarray:
    """
    Example:
        data: b"\xf0"
        return: np.array([True, True, True, True, False, False, False, False])
    """
    uint8_arr = np.frombuffer(data, dtype=np.uint8)
    bit_arr = np.unpackbits(uint8_arr).astype(bool)
    return bit_arr


def int_to_bits(x: int, length: int) -> np.ndarray:
    """
    Example 1:
        x: 0b11110000
        length: 8
        return: np.array([True, True, True, True, False, False, False, False])

    Example 2:
        x: 0b11110000
        length: 9
        return: np.array([False, True, True, True, True, False, False, False, False])
    """
    b = x.to_bytes((length + 7) // 8, byteorder="big")
    uint8_arr = np.frombuffer(b, dtype=np.uint8)
    bit_arr = np.unpackbits(uint8_arr).astype(bool)
    return bit_arr[-length:]


def int_to_bits_str(x: int, length: int) -> str:
    """
    Example:
        x: 0b110
        length: 6
        return: "000110"
    """
    bits = int_to_bits(x, length)
    return "".join("1" if b else "0" for b in bits)


def count_query(data: torch.Tensor, query: torch.Tensor) -> int:
    """
    Example:
        data: torch.tensor([True, True, False, True, False, True, False])
        query: torch.tensor([True, False, True])
        return: 2
    Note: this is a convolutional check
    """
    q1 = query
    q2 = ~query
    expect1 = q1.sum().item()
    expect2 = 0
    r1 = torch.conv1d(data.view(1, 1, -1).float(), q1.view(1, 1, -1).float()).view(-1) == expect1
    r2 = torch.conv1d(data.view(1, 1, -1).float(), q2.view(1, 1, -1).float()).view(-1) == expect2
    return (r1 & r2).sum().item()


def plot_results(results: dict[int, int], n: int, save_folder: str = None, filename: str = None) -> None:
    """
    Example Inputs:
        results: {0b00: 1, 0b01: 2, 0b10: 3, 0b11: 4}
        n: 2
    """
    x, y = zip(*results.items())
    x = list(map(lambda i: int_to_bits_str(i, n), x))
    plt.clf()
    plt.figure(figsize=(len(x) // 5, 5))
    plt.bar(x, y)
    plt.xticks(rotation=-90)

    if save_folder is not None and filename is not None:
        plt.gcf().set_dpi(100)
        plt.savefig(os.path.join(save_folder, filename))
    else:
        plt.show()


def save_results_csv(results: dict[int, int], n: int, save_folder: str, filename: str) -> None:
    """
    Example Inputs:
        results: {0b00: 1, 0b01: 2, 0b10: 3, 0b11: 4}
        n: 2
        title: "Example"
    """
    x, y = zip(*results.items())
    x = list(map(lambda i: int_to_bits_str(i, n), x))
    with open(os.path.join(save_folder, filename), "w") as file:
        file.write("pattern, count\n")
        for i, j in zip(x, y):
            file.write(f"{i}, {j}\n")



if __name__ == "__main__":
    data_folder = "./data"
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)

    for filename in sorted(os.listdir(data_folder, )):
        if filename.startswith("."):
            print(f"Skipping `{filename}`...")
            continue
        filename = os.path.join(data_folder, filename)
        if not os.path.isfile(filename):
            continue
        print(f"Processing `{filename}`...")

        with open(filename, "rb") as file:
            data = file.read()

        MAX_BYTES = 1_000_000
        data = data[:MAX_BYTES]

        bits_arr = bytes_to_bits(data)
        bits_arr = torch.tensor(bits_arr, dtype=torch.bool)

        for n in [4, 8]:
            results = {i: 0 for i in range(2**n)}
            for i in trange(2**n):
                query = torch.tensor(int_to_bits(i, n), dtype=torch.bool)
                count = count_query(bits_arr, query)
                results[i] = count
            # save_results_csv(results, n, results_folder, f"{file_path.split('/')[-1]}, n={n}.csv")
            plot_results(results, n, results_folder, f"{filename.split('/')[-1]}, n={n}.png")

