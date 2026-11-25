import csv
import zipfile
from pathlib import Path

TEXT_EXTENSIONS = [
    ".txt",
    ".py",
    ".js",
    ".json",
    ".md",
    ".html",
    ".css",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
    ".sh",
]
SPREADSHEET_EXTENSIONS = [".xlsx", ".xls", ".csv"]


def _read_text_file(file_path: str, start_line: int = 0, end_line: int = -1) -> str:
    """Read UTF-8 text and render a selected line range with line numbers.

    Args:
        file_path: Path to the text file.
        start_line: 1-based first line; zero also selects the first line and labels
            it zero.
        end_line: Inclusive last line, or ``-1`` to read through end-of-file.

    Returns:
        Selected lines formatted as ``line-number | text``.

    Raises:
        OSError: If the file cannot be opened.
        UnicodeDecodeError: If the contents are not valid UTF-8.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Adjust line numbers (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))

    selected_lines = lines[start_idx:end_idx]

    result = []
    for i, line in enumerate(selected_lines, start=start_line):
        result.append(f"{i:4d} | {line.rstrip()}")
    return "\n".join(result)


def _read_csv(file_path: str) -> str:
    """Read a CSV file and format it as an aligned Markdown-style table.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A table containing the first row as the header and remaining rows as data,
        or an empty string for a file with no rows.

    Raises:
        OSError: If the file cannot be opened.
        csv.Error: If CSV parsing fails.
    """
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if not rows:
        return ""

    column_count = max(len(row) for row in rows)
    rows = [row + [""] * (column_count - len(row)) for row in rows]
    widths = [max(len(row[column]) for row in rows) for column in range(column_count)]

    def format_row(row: list[str]) -> str:
        """Pad each cell to its column width and add table delimiters.

        Args:
            row: Values from one CSV row, already padded to the table width.

        Returns:
            One aligned table row as a string.
        """
        return (
            "| "
            + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(rows[0]), separator, *map(format_row, rows[1:])])


def _read_excel(file_path: str) -> str:
    """Read the first worksheet of an Excel file into a text table.

    Args:
        file_path: Path to an ``.xlsx`` or ``.xls`` workbook.

    Returns:
        The first worksheet rendered as a string without row indices.

    Raises:
        RuntimeError: If pandas or a compatible Excel reader is unavailable.
        ImportError: If pandas cannot find a compatible Excel engine.
        OSError, ValueError: If the workbook cannot be read or parsed.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError(
            "Reading Excel files requires pandas and an Excel reader such as openpyxl."
        ) from error

    return pd.read_excel(file_path).to_string(index=False)


def unzip_file(zip_path: str, extract_to: str = None) -> str:
    """Extract every member of a ZIP archive and summarize its contents.

    Args:
        zip_path: Archive path.
        extract_to: Destination directory; defaults to a sibling folder named
            after the archive without its extension.

    Returns:
        An error string if the archive is missing; otherwise an extraction count,
        destination, and up to twenty member names.

    Raises:
        zipfile.BadZipFile: If the archive is corrupt or not a ZIP file.
        OSError: If extraction cannot create or write destination files.

    Notes:
        This function does not perform path-traversal filtering; callers should
        only extract trusted archives.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return f"File not found: {zip_path}"

    # Default extraction path: create folder with zip filename
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    else:
        extract_to = Path(extract_to)

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extractall(extract_to)

    # Format results
    result = f"Extracted {len(file_list)} files to {extract_to}/\n\n"
    result += "Contents:\n"
    for f in file_list[:20]:
        result += f"  - {f}\n"
    if len(file_list) > 20:
        result += f"  ... and {len(file_list) - 20} more files\n"

    return result


def list_files(path: str = ".") -> str:
    """List visible children of a directory, with directories first.

    Args:
        path: Directory path to list.

    Returns:
        A formatted directory listing, or a message if the path is missing or is
        not a directory. Dot-prefixed entries are omitted.

    Raises:
        OSError: If directory enumeration fails.
    """
    path = Path(path)

    if not path.exists():
        return f"Path not found: {path}"

    if not path.is_dir():
        return f"Not a directory: {path}"

    items = []
    for item in sorted(path.iterdir()):
        if item.name.startswith("."):
            continue

        if item.is_dir():
            items.append(f"{item.name}/")
        else:
            items.append(f"{item.name}")

    # Sort directories first
    dirs = [i for i in items if i.endswith("/")]
    files = [i for i in items if not i.endswith("/")]

    result = f"Directory: {path}\n"
    for item in dirs + files:
        result += f"  {item}\n"

    return result


def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read a supported file as numbered text, a table, or spreadsheet output.

    Args:
        file_path: Path to the file.
        start_line: First 1-based line for text files; ignored for tabular files.
        end_line: Inclusive final text line, or ``-1`` for the rest of the file;
            ignored for tabular files.

    Returns:
        Numbered text, a CSV table, an Excel table, or a missing-file message.
        Unknown extensions are attempted as UTF-8 text.

    Raises:
        RuntimeError: If Excel support dependencies are unavailable.
        OSError: If an existing file cannot be opened.
        UnicodeDecodeError: If text contents are not valid UTF-8.
    """
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path, start_line, end_line)
    elif ext == ".csv":
        return _read_csv(file_path)
    elif ext in SPREADSHEET_EXTENSIONS:
        return _read_excel(file_path)
    else:
        return _read_text_file(file_path, start_line, end_line)
