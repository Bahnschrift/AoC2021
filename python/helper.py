from __future__ import annotations
# from _typeshed import SupportsGreaterThan, SupportsLessThan  # For Python < 3.10

import math
import os
import re
from itertools import chain, combinations, product
from typing import Any, Iterable, Sized, Callable, Sequence, Iterator, TypeVar, Generic, TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import SupportsGreaterThan, SupportsLessThan  # For Python < 3.10
from pythonlangutil.overload import Overload, signature

import requests
from bs4 import BeautifulSoup
from rich import print

ADJACENT_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ADJACENT_8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

COOKIE_PATH = "../inputs/cookie.txt"


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
def _get_session_cookie() -> str | None:
    """Get the session cookie from the cookie file.

    :returns: The session cookie.
    """
    try:
        with open(COOKIE_PATH, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        _update_session_cookie()
        _get_session_cookie()


def _update_session_cookie() -> None:
    """Update the session cookie in the cookie file."""
    print("[bold yellow]Updating session cookie...")
    with open(COOKIE_PATH, "w+") as f:
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
    assert soup.article is not None
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
        inp = f.read().rstrip("\n")
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


def get_input_ints(day: int, year: int = 2021, path: str = "../inputs/") -> list[int]:
    """Gets the AoC input as a list of ints.

    :param day: the day of the AoC challenge
    :param year: the year of the AoC challenge (default: 2021)
    :param path: the path to the folder containing the input file (default: "../inputs/")
    :returns: the input as a list of ints
    """
    return list(map(int, get_input_lines(day=day, year=year, path=path)))


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


def rotate_grid(grid: list[list], n: int = 1) -> list[list]:
    """Rotates a 2D list by 90 degrees clockwise n times.

    Example:
    >>> rotate_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], n=1)
    [[7, 4, 1], [8, 5, 2], [9, 6, 3]]

    :param grid: The grid to rotate
    :param n: The number of times to rotate the grid (default: 1)
    :returns: The rotated grid
    """
    if n < 0:
        return rotate_grid(grid, n + 4)
    if n > 3:
        return rotate_grid(grid, n - 4)
    for _ in range(n):
        grid = [[grid[y][x] for y in range(len(grid) - 1, -1, -1)] for x in range(len(grid[0]))]
    return grid


def flip_grid_y(grid: list[list]) -> list[list]:
    """flips a 2D list across the y-axis.

    Example:
    >>> flip_grid_y([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [[3, 2, 1], [6, 5, 4], [9, 8, 7]]

    :param grid: the 2D list to flip
    :returns: the flipped 2D list
    """
    return [line[::-1] for line in grid]


def flip_grid_x(grid: list[list]) -> list[list]:
    """flips a 2D list across the x-axis.

    Example:
    >>> flip_grid_x([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    :param grid: the 2D list to flip
    :returns: the flipped 2D list
    """
    return grid[::-1]


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
    """Flips (inverts) the ith bit of n.

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
    >>> poweriter((1, 2, 2))
    [(), (1,), (2,), (2,), (1, 2), (1, 2), (2, 2), (1, 2, 2)]

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


# +----------------------------------------------------------------------------+
# |                                                                            |
# |                               The Grid                                     |
# |                                                                            |
# +----------------------------------------------------------------------------+



T = TypeVar("T")

class Grid(Generic[T]):
    """A helpful class for dealing with 2D arrays.
    Initialize with either a 2D list or a default value, width and height.
    """

    @staticmethod
    def of2Dlist(l: list[list[T]]) -> Grid:
        """Initializes a Grid with a 2D list.

        :param l: the 2D list to initialize the grid with
        :returns: the grid initialized with the 2D list
        """
        return Grid(l)

    @staticmethod
    def ofValWidthHeight(val: T, width: int, height: int) -> Grid:
        """Initializes a grid with a default value and width and height.
        [[0, 0], [0, 0]]

        :param val: the default value
        :param width: the width of the grid
        :param height: the height of the grid
        :returns: a grid with the specified width and height and the default value
        """
        return Grid([[val] * width for _ in range(height)])

    def __init__(self, grid: Sequence[Sequence[T]], default: Any = None):
        # T = TypeVar("T")
        self.grid: list[list[T]]
        self.width: int
        self.height: int

        if len(grid) == 0:
            raise ValueError("Grid cannot be empty.")

        if all(len(row) == len(grid[0]) for row in grid):
            self.grid = [[*row] for row in grid]
        elif default is not None:
            self.width = len(max(grid, key=len))
            self.grid = [[*row] + [default] * (self.width - len(row)) for row in grid]
        else:
            raise ValueError("Grid must be rectangular or a default value must be given to pad shorter rows.")

        self.width = len(grid[0])
        self.height = len(grid)

    @Overload
    @signature("tuple")
    def __getitem__(self, pos: tuple[int, int]) -> T:
        if not (0 <= pos[0] < self.width):
            raise IndexError(f"X index {pos[0]} out of range.")
        elif not (0 <= pos[1] < self.height):
            raise IndexError(f"Y index {pos[1]} out of range.")
        return self.grid[pos[1]][pos[0]]
    
    @__getitem__.overload
    @signature("int")
    def __getitem__(self, pos: int) -> list[T]:
        return self.grid[pos]
    
    @__getitem__.overload
    @signature("slice")
    def __getitem__(self, pos: slice) -> Grid[T]:
        return Grid(self.grid[pos])

    def __setitem__(self, pos: tuple[int, int], value: Any):
        if not (0 <= pos[0] < self.height):
            raise IndexError(f"Y index {pos[0]} out of range.")
        elif not (0 <= pos[1] < self.width):
            raise IndexError(f"X index {pos[1]} out of range.")
        self.grid[pos[1]][pos[0]] = value

    def __str__(self) -> str:
        return "\n".join(map(str, self.grid))

    def __repr__(self) -> str:
        return f"Grid({self.grid})"

    def __len__(self) -> int:
        return self.height

    def __contains__(self, item) -> bool:
        return any(item in row for row in self.grid)

    def __iter__(self) -> Iterable[list]:
        return iter(self.grid)

    def __eq__(self, other):
        return (
            self.grid == other.grid
            or self.width == other.width
            and self.height == other.height
            and all(self[x, y] == other[x, y] for x in range(self.height) for y in range(self.width))
        )

    def __add__(self, other: Grid[T]) -> Grid:
        """Adds the items of two grids together."""
        new = []
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must be the same size.")
        for y, row in enumerate(self.grid):
            r = []
            for x, cell in enumerate(row):
                r.append(cell + other[x, y])
            new.append(r)
        return Grid(new)

    def __sub__(self, other: Grid[T]) -> Grid:
        """Subtracts the items of two grids."""
        new = []
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must be the same size.")
        for y, row in enumerate(self.grid):
            r = []
            for x, cell in enumerate(row):
                new = cell - other[x, y]
            new.append(r)
        return Grid(new)

    def __mul__(self, other: Grid):
        """Multiplies the items of two grids."""
        new = []
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must be the same size.")
        for y, row in enumerate(self.grid):
            r = []
            for x, cell in enumerate(row):
                r.append(cell * other[x, y])
            new.append(r)
        return Grid(new)

    def __truediv__(self, other: Grid):
        """Divides the items of two grids."""
        new = []
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must be the same size.")
        for y, row in enumerate(self.grid):
            r = []
            for x, cell in enumerate(row):
                r.append(cell / other[x, y])
            new.append(r)
        return Grid(new)

    def __floordiv__(self, other: Grid):
        """Divides the items of two grids using the floor function."""
        new = []
        if self.width != other.width or self.height != other.height:
            raise ValueError("Grids must be the same size.")
        for y, row in enumerate(self.grid):
            r = []
            for x, cell in enumerate(row):
                r.append(cell // other[x, y])
            new.append(r)
        return Grid(new)

    def str_aligned_l(self, sep: str = ", ", pad: str = " ") -> str:
        """Gets a string representation of the grid with each columned padded to a uniform length.
        Shorter items are left-aligned."""
        mapped = self.mapped_items(str)
        widths = [len(max(row, key=len)) for row in mapped.rotated()]
        return "\n".join(sep.join(item.ljust(widths[x], pad) for x, item in enumerate(row)) for row in mapped)

    def str_aligned_r(self, sep: str = ", ", pad: str = " ") -> str:
        """Gets a string representation of the grid with each columned padded to a uniform length.
        Shorter items are right-aligned."""
        mapped = self.mapped_items(str)
        widths = [len(max(row, key=len)) for row in mapped.rotated()]
        return "\n".join(sep.join(item.rjust(widths[x], pad) for x, item in enumerate(row)) for row in mapped)

    def copy(self):
        return Grid(self.grid)

    def size(self) -> int:
        return self.width * self.height

    def row(self, index: int) -> list:
        """Gets the row at the given index from the grid."""
        return self.grid[index]

    def col(self, index: int) -> list:
        """Gets the column at the given index from the grid."""
        return [row[index] for row in self.grid]

    def pos_exists(self, pos: tuple[int, int]) -> bool:
        """Checks if the given position exists in the grid."""
        return 0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def positions(self) -> Iterator[tuple[int, int]]:
        """Returns an iterator of all positions in the grid."""
        return product(range(self.height), range(self.width))

    def contains(self, item: Any) -> bool:
        """Checks if the given item is in the grid."""
        return self.__contains__(item)

    def append(self, row: Sequence) -> None:
        """Appends the given row to the grid."""
        if len(row) != self.width:
            raise ValueError(f"Row length must match grid width ({len(row)} != {self.width}).")
        self.grid.append(list(row))
        self.height += 1

    def append_row(self, row: Sequence) -> None:
        """Appends the given row to the grid (alias for Grid.append)."""
        self.append(row)

    def append_col(self, col: Sequence) -> None:
        """Appends the given column to the grid."""
        if len(col) != self.height:
            raise ValueError(f"Column must be the same length as the grid ({len(col)} != {self.height}).")
        self.grid = [row + [col[i]] for i, row in enumerate(self.grid)]
        self.width += 1

    def prepend(self, row: Sequence) -> None:
        """Prepends the given row to the grid."""
        if len(row) != self.width:
            raise ValueError(f"Row length must match grid width ({len(row)} != {self.width}).")
        self.grid = [list(row)] + self.grid
        self.height += 1

    def prepend_row(self, row: Sequence) -> None:
        """Prepends the given row to the grid (alias for Grid.prepend)."""
        self.prepend(row)

    def prepend_col(self, col: Sequence) -> None:
        """Prepends the given column to the grid."""
        if len(col) != self.height:
            raise ValueError(f"Column must be the same length as the grid ({len(col)} != {self.height}).")
        self.grid = [[col[i]] + row for i, row in enumerate(self.grid)]
        self.width += 1

    def pop_row(self, index: int = -1) -> list:
        """Pops the row at the given index from the grid."""
        self.height -= 1
        row = self.grid.pop(index)
        return row

    def pop_col(self, index: int = -1) -> list:
        """Pops the column at the given index from the grid."""
        self.width -= 1
        col = [row.pop(index) for row in self.grid]
        return col

    def get_pos(self, item: Any) -> tuple[int, int]:
        """Returns the first found position of the given item in the grid in the format (x, y)."""
        for y, row in enumerate(self.grid):
            for x, col in enumerate(row):
                if col == item:
                    return y, x
        raise ValueError(f"{item} is not in grid.")

    def replace(self, old: Any, new: Any, count: int = None) -> None:
        """Replaces all instances of the old item with the new item."""
        count = count or len(self)
        for y, row in enumerate(self.grid):
            for x, col in enumerate(row):
                if col == old:
                    self[x, y] = new
                    count -= 1
                    if count == 0:
                        return

    def replaced(self, old: Any, new: Any, count: int = None) -> Grid:
        """Returns a copy of the grid with all instances of the old item replaced with the new item."""
        grid = self.copy()
        grid.replace(old, new, count)
        return grid

    def count(self, item: Any) -> int:
        """Counts the number of times the given item is in the grid."""
        return sum(row.count(item) for row in self.grid)

    def max(self, key: Callable[[T], SupportsGreaterThan]=None):
        """Finds the item of the largest value in the grid."""
        return max((max(row, key=key) for row in self.grid), key=key)

    def min(self, key: Callable[[T], SupportsLessThan] = None):
        """Finds the item of the smallest value in the grid."""
        return min((min(row, key=key) for row in self.grid), key=key)

    def rotate(self, n: int = 1) -> None:
        """Rotates the grid n times."""
        self.grid = rotate_grid(self.grid, n)

    def rotated(self, n: int = 1) -> Grid[T]:
        """Returns a rotated copy of the grid."""
        return Grid(rotate_grid(self.grid, n))

    def flip_y(self) -> None:
        """Flips the grid."""
        self.grid = flip_grid_y(self.grid)

    def flipped_y(self) -> Grid[T]:
        """Returns a flipped copy of the grid."""
        return Grid(flip_grid_y(self.grid))

    def flip_x(self) -> None:
        """Flips the grid."""
        self.grid = flip_grid_x(self.grid)

    def flipped_x(self) -> Grid[T]:
        """Returns a flipped copy of the grid."""
        return Grid(flip_grid_x(self.grid))

    def rotations(self) -> Iterator[Grid]:
        """Returns an iterator of all rotations of the grid."""
        for i in range(4):
            yield self.rotated(i)

    def permutations(self) -> Iterator[Grid[T]]:
        """Returns an iterator of all the possible rotations (including mirrored) of the grid."""
        for rot in self.rotations():
            yield rot
            yield rot.flipped_y()

    def edges(self) -> Iterator[list[T]]:
        """Returns an iterator of all four edges of the grid."""
        yield self.row(0)
        yield self.col(0)
        if self.height > 1:
            yield self.row(-1)
        if self.width > 1:
            yield self.col(-1)

    def edge_positions(self) -> Iterator[tuple[int, int]]:
        """Returns an iterator of all the positions of each tile on the edge."""
        for x in range(self.width):
            for y in range(self.height):
                if x == 0 or y == 0 or x == self.width - 1 or y == self.height - 1:
                    yield x, y

    N = TypeVar("N")

    def map_items(self, func: Callable[[T], N]) -> None:
        """Applies a function to every item in the grid."""
        self.grid = [[*map(func, row)] for row in self.grid]

    def mapped_items(self, func: Callable[[T], N]) -> Grid[N]:
        """Returns a copy of the grid with a function applied to every item."""
        new = self.copy()
        new.map_items(func)
        return new

    def map_rows(self, func: Callable[[list[T]], list[N]]) -> None:
        """Applies a function to every row in the grid."""
        self.grid = [*map(func, self.grid)]

    def mapped_rows(self, func: Callable[[list[T]], list[N]]) -> Grid[N]:
        """Returns a copy of the grid with a function applied to every row."""
        new = self.copy()
        new.map_rows(func)
        return new

    def map_cols(self, func: Callable[[list[T]], list[N]]) -> None:
        """Applies a function to every column in the grid."""
        self.grid = self.rotated(-1).mapped_rows(func).rotated(1)

    def mapped_cols(self, func: Callable[[list[T]], list[N]]) -> Grid[N]:
        """Returns a copy of the grid with a function applied to every column."""
        # TODO: This is a bit inefficient. Make it better.
        new = self.copy()
        new.map_cols(func)
        return new

    def neighbours_pos(self, pos: tuple[int, int]) -> Iterator[tuple[int, int]]:
        """Returns an iterator of all the positions of the neighbours of each tile."""
        x, y = pos
        for dx, dy in ADJACENT_4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield nx, ny
    
    def neighbours(self, pos: tuple[int, int]) -> Iterator[T]:
        """Returns an iterator of all the neighbours of each tile."""
        for x, y in self.neighbours_pos(pos):
            yield self[x, y]
    