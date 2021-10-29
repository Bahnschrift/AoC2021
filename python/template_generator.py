import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy the template file for the given days")
    parser.add_argument("-s", "--start", type=int, help="Starting Day (inclusive)", required=False)
    parser.add_argument("-e", "--end", type=int, help="Ending Day (inclusive", required=True)
    parser.add_argument("-o", "--overwrite", type=bool, help="Overwrite existing files", required=False)

    args = parser.parse_args()
    start = args.start
    end = args.end
    overwrite = args.overwrite
    if start is None:
        start = 1

    print(
        f"Copying template for days {start} to {end} (inclusive){' with overwrite mode enabled' if overwrite else ''}.")
    if input("Press enter to continue..."):
        exit("Cancelling.")

    for day in range(start, end + 1):
        with open("template.ipynb", "r") as template_file:
            template = template_file.read()
            template = template.replace(" 1", f" {day}").replace("/1", f"/{day}")
            new_name = f"day{day}.ipynb"
            if overwrite or not os.path.isfile(new_name):
                with open(new_name, "w+") as new_file:
                    new_file.write(template)
            else:
                print(f"Skipping {new_name} as it already exists.")
