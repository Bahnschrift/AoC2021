import math
import os
import re
from itertools import chain, combinations
from typing import Any, Iterable, Sized

import requests
from bs4 import BeautifulSoup
from rich import print

ADJACENT_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ADJACENT_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


# +----------------------------------------------------------------------------+
# |                                                                            |
# |                            Output formatting                               |
# |                                                                            |
# +----------------------------------------------------------------------------+
def _print_part(n: int, ans: Any) -> None:
    """Print answer for part n.

    :param n: the part to print
    :param ans: the answer to print
    """
    print(f"[bold green]Part {n}:[/] {ans}")


def print_part_1(ans: Any) -> None:
    """Print answer for part 1.

    :param ans: the answer to print
    """
    _print_part(1, ans)


def print_part_2(ans: Any) -> None:
    """Print answer for part 2.

    :param ans: the answer to print
    """
    _print_part(2, ans)


# +----------------------------------------------------------------------------+
# |                                                                            |
# |                            Submitting answers                              |
# |                                                                            |
# +----------------------------------------------------------------------------+
def _get_session_cookie() -> str:
    """Get the session cookie from the cookie file.

    :returns: The session cookie.
    """
    try:
        with open("../inputs/cookie.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        _update_session_cookie()
        _get_session_cookie()


def _update_session_cookie() -> None:
    """Update the session cookie in the cookie file."""
    print("[bold yellow]Updating session cookie...")
    with open("../inputs/cookie.txt", "w+") as f:
        new = input("Session cookie is invalid. Please enter the new session cookie or leave blank to cancel")
        if not new:
            exit(1)
        f.write(new)
    print("[bold green]Session cookie updated.")


def _submit(x: int, ans: Any, day: int, year: int) -> None:
    """Submit answer for part x.

    :param x: the part to _submit
    :param ans: the answer to _submit
    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    """
    headers = {"session": _get_session_cookie()}
    url = f"https://adventofcode.com/{year}/day/{day}/answer"
    data = {"level": x, "answer": str(ans)}
    response = requests.post(url, cookies=headers, data=data)
    soup = BeautifulSoup(response.text, "html.parser")
    message = soup.article.text
    response.close()

    if message.startswith("--- Day"):
        _update_session_cookie()
        _submit(x, ans, day, year)
        return

    print(f"[bold]Part {x} submission:")
    if message.startswith("That's the right answer!"):
        print(f"[bold green]Correct answer.")
    elif message.startswith("That's not the right answer"):
        if "too low" in message:
            print(f" [bold red]Too low.")
        elif "too high" in message:
            print(f" [bold red]Too high.")
        else:
            print(f"[bold red]Incorrect answer.")
    elif message.startswith("You don't seem to be solving the right level"):
        print(f"[bold yellow]Already solved{' or part 1 not yet completed' if x == 2 else ''}.")
    elif message.startswith("You gave an answer too recently"):
        print(f"[bold yellow]Too fast. {message.split('.')[1].strip()}.")
    else:
        print(f"[bold red]Unknown response:[/]")
        print(message)


def submit_part_1(ans: Any, day: int, year: int = 2021) -> None:
    """Submit answer for part 1.

    :param ans: the answer to _submit
    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    """
    _submit(1, ans, day, year)


def submit_part_2(ans: Any, day: int, year: int = 2021) -> None:
    """Submit answer for part 2.

    :param ans: the answer to _submit
    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    """
    _submit(2, ans, day, year)


# +----------------------------------------------------------------------------+
# |                                                                            |
# |                            Input downloading                               |
# |                                                                            |
# +----------------------------------------------------------------------------+
def _download_input(day: int, year: int, path: str) -> None:
    """Download the input file and save it to the given path.

    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    :param path: the path to the folder containing the input file (default: "../inputs/")
    """
    headers = {"session": _get_session_cookie()}
    url = f"https://adventofcode.com/{year}/day/{day}/input"
    request = requests.post(url, cookies=headers)
    text = request.text
    request.close()

    if text.startswith("Puzzle inputs differ by user."):
        _update_session_cookie()
        _download_input(day, year, path)
    elif text.startswith("Please don't repeatedly request this endpoint"):
        print("[bold red]Failed to download input, please try again later.")
        text = ""
    else:
        print("[bold green]Input download successful.")
    with open(f"{path}/day{day}.txt", "w+") as f:
        f.write(text)


def _ensure_input_file(day: int, year: int, path: str) -> None:
    """Ensure the input file exists and contains data.

    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    :param path: the path to the folder containing the input file (default: "../inputs/")
    """
    if f"day{day}.txt" not in os.listdir(path) or open(f"{path}/day{day}.txt", "r").read() == "":
        print(f"[italic]Downloading input for day {day}...")
        _download_input(day, year, path)


def get_input(day: int, year: int = 2021, path: str = "../inputs/") -> str:
    """Gets the AoC input as a string.

    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    :param path: the path to the folder containing the input file (default: "../inputs/")
    :returns: the input as a string.
    """
    _ensure_input_file(day, year, path)
    with open(f"{path}/day{day}.txt", "r") as f:
        inp = f.read()
    return inp


def get_input_lines(day: int, year: int = 2021, path: str = "../inputs/") -> list[str]:
    """Gets the AoC input as a list of lines.

    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    :param path: the path to the folder containing the input file (default: "../inputs/")
    :returns: the input as a list of lines
    """
    _ensure_input_file(day, year, path)
    with open(f"{path}/day{day}.txt", "r") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]
    return lines


# +----------------------------------------------------------------------------+
# |                                                                            |
# |                            Misc. Functions                                 |
# |                                                                            |
# +----------------------------------------------------------------------------+
def digits(s: str) -> list[int]:
    """Finds all single digits in a string.

    Example:
    >>> digits("1a23.5")
    [1, 2, 3, 5]

    :param s: The string to find digits in.
    :returns: A list of all digits in the string.
    """
    return [int(x) for x in s if x.isdigit()]


def ints(s: str) -> list[int]:
    """Finds all integers in a string.

    Example:
    >>> ints("1a23.5")
    [1, 23, 5]

    :param s: the string to find integers in
    :returns: a list of all integers in the string
    """
    return [int(x) for x in re.findall(r"-?\d+", s)]


def floats(s: str) -> list[float]:
    """Finds all floats in a string.

    Example:
    >>> floats("1a-23.5")
    [1.0, -23.5]

    :param s: the string to find floats in
    :returns: a list of all floats in the string
    """
    return [*map(float, re.findall(r"([\-+]?\d*(?:\d|\d\.|\.\d)\d*)", s))]


def rotate_grid(grid: list[list[int]], n: int = 1) -> list[list[int]]:
    """Rotates a 2D list by 90 degrees clockwise n times.

    Example:
    >>> rotate_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], n=1)
    [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    :param grid: The grid to rotate
    :param n: The number of times to rotate the grid (default: 1)
    :returns: The rotated grid
    """
    grid = grid[:]
    for _ in range(n):
        grid = list(zip(*grid[::-1]))
    return grid


def flip_grid(grid: list[list[int]]) -> list[list[int]]:
    """Flips a 2D list across the y-axis.

    Example:
    >>> flip_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [[3, 2, 1], [6, 5, 4], [9, 8, 7]]

    :param grid: the 2D list to flip
    :returns: the flipped 2D list
    """
    return [line[::-1] for line in grid]


def get_bit(n: int, i: int) -> int:
    """Gets the ith bit of n.

    Example:
    >>> get_bit(0b1101, 2)
    1

    :param n: the number to get the bit from
    :param i: the index of the bit to get
    :returns: the ith bit of n
    """
    return (n >> i) & 1


def set_bit(n: int, i: int, v: int) -> int:
    """Sets the ith bit of n to v.

    Example:
    >>> set_bit(0b1101, 1, 1)
    0b1101

    :param n: the number to set the bit in
    :param i: the index of the bit to set
    :param v: the value to set the bit to
    :returns: the number with the ith bit set to v
    """
    return n | (v << i)


def flip_bit(n: int, i: int) -> int:
    """Flips the ith bit of n.

    Example:
    >>> flip_bit(0b1101, 2)
    0b1001

    :param n: the number to flip the bit in
    :param i: the index of the bit to flip
    :returns: the number with the ith bit flipped
    """
    return n ^ (1 << i)


def reverse_bits(n: int) -> int:
    """Reverses the bits of n.

    Example:
    >>> reverse_bits(0b1101)
    0b1011

    :param n: the number to reverse the bits of
    :returns: the number with its bits reversed
    """
    return int(f"{n:b}"[::-1], 2)


def reverse_dictionary(d: dict) -> dict:
    """Reverses a dictionary.

    Example:
    >>> reverse_dictionary({'a': 1, 'b': 2})
    {1: 'a', 2: 'b'}

    :param d: the dictionary to reverse
    :returns: the reversed dictionary
    """
    return {v: k for k, v in d.items()}


def powerset(s: set) -> list[set]:
    """Finds the powerset of a set.

    Example:
    >>> powerset({1, 2, 3})
    {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}

    :param s: the set to find the powerset of
    :returns: the powerset of s
    """
    return list(map(set, chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))))


def poweriter(s: Iterable) -> list[tuple]:
    """Finds the powerset of any iterable.

    Example:
    >>> poweriter({1, 2, 3})
    {{}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}}

    :param s: the iterable to find the powerset of
    :returns: the powerlist of s
    """
    try:
        assert isinstance(s, Sized) and isinstance(s, Iterable)
    except AssertionError:
        raise TypeError(f"{type(s)} does not define the method __{'len' if isinstance(s, Iterable) else 'iter'}__.")
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def flatten(ls: Iterable[Iterable]) -> list:
    """Flattens a 2D list.

    Example:
    >>> flatten([[1, 2], [3, 4]])
    [1, 2, 3, 4]

    :param ls: the list to flatten
    :returns: the flattened list
    """
    return [item for sublist in ls for item in sublist]


def full_flatten(ls: Iterable) -> list:
    """Flattens a list of any depth.

    Example:
    >>> full_flatten([1, [2, [3, [4]]]])
    [1, 2, 3, 4]

    :param ls: the list to flatten
    :returns: the flattened list
    """
    flattened = []
    for el in ls:
        if isinstance(el, Iterable) and not isinstance(el, str):
            flattened += full_flatten(el)
        else:
            flattened += [el]
    return flattened


def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Gets the distance between two points.

    Example:
    >>> distance((0, 0), (3, 4))
    5.0

    :param p1: the first point
    :param p2: the second point
    :returns: the distance between p1 and p2
    """
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distance_manhattan(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Gets the manhattan distance between two points.

    Example:
    >>> distance_manhattan((1, 2), (3, 4))
    4

    :param p1: the first point
    :param p2: the second point
    :returns: the manhattan distance between p1 and p2
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def dot_product(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Gets the dot product of two 2D vectors.

    Example:
    >>> dot_product((1, 2), (3, 4))
    11

    :param p1: the first vector
    :param p2: the second vector
    :returns: the dot product of p1 and p2
    """
    return p1[0] * p2[0] + p1[1] * p2[1]


def dot_iter(p1: Iterable[float], p2: Iterable[float]) -> float:
    """Gets the dot product of two nD vectors.

    Example:
    >>> dot_iter([1, 2, 3, 4], [5, 6, 7, 8])
    70

    :param p1: the first vector
    :param p2: the second vector
    :returns: the dot product of p1 and p2
    """
    return sum(map(lambda x, y: x * y, p1, p2))


def sign(n: int) -> int:
    """Gets the sign of a number.

    Example:
    >>> sign(1)
    1

    :param n: the number to get the sign of
    :returns: the sign of n
    """
    return 0 if n == 0 else 1 if n > 0 else -1
