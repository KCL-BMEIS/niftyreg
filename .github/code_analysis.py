import os
import re
import subprocess
import argparse
import sys
from github import Github

# Input variables from Github action
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PR_NUM = os.getenv("PR_NUMBER", "-1")
WORK_DIR = f'{os.getenv("GITHUB_WORKSPACE")}'
REPO_NAME = os.getenv("REPO")
TARGET_REPO_NAME = os.getenv("REPO", "")
SHA = os.getenv("GITHUB_SHA")
COMMENT_TITLE = os.getenv("COMMENT_TITLE", "Static Analysis")
ONLY_PR_CHANGES = os.getenv("REPORT_PR_CHANGES_ONLY", "False").lower()
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
FILES_WITH_ISSUES = {}

# Max characters per comment - 65536
# Make some room for HTML tags and error message
MAX_CHAR_COUNT_REACHED = "!Maximum character count per GitHub comment has been reached! Not all warnings/errors has been parsed!"
COMMENT_MAX_SIZE = 65000
CURRENT_COMMENT_LENGTH = 0


def debug_print(message):
    if VERBOSE:
        lines = message.split("\n")
        for line in lines:
            print(f"\033[96m {line}")


def parse_diff_output(changed_files):
    """
    Parses the diff output to extract filenames and corresponding line numbers of changes.

    The function identifies changed lines in files and excludes certain directories
    based on the file extension. It then extracts the line numbers of the changes
    (additions) and associates them with their respective files.

    Parameters:
    - changed_files (str): The diff output string.

    Returns:
    - dict: A dictionary where keys are filenames and values are lists of line numbers
            that have changes.

    Usage Example:
    ```python
    diff_output = "<output from `git diff` command>"
    changed_file_data = parse_diff_output(diff_output)
    for file, lines in changed_file_data.items():
        print(f"File: {file}, Changed Lines: {lines}")
    ```

    Note:
    - The function only considers additions in the diff, lines starting with "+".
    - Filenames in the return dictionary include their paths relative to the repo root.
    """

    # Regex to capture filename and the line numbers of the changes
    file_pattern = re.compile(r"^\+\+\+ b/(.*?)$", re.MULTILINE)
    line_pattern = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

    supported_extensions = (".h", ".hpp", ".hcc", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cl")

    files = {}
    for match in file_pattern.finditer(changed_files):
        file_name = match.group(1)

        # Filtering for language specific files and excluding certain directories
        if file_name.endswith(supported_extensions):
            # Find the lines that changed for this file
            lines_start_at = match.end()
            next_file_match = file_pattern.search(changed_files, pos=match.span(0)[1])

            # Slice out the part of the diff that pertains to this file
            file_diff = changed_files[lines_start_at : next_file_match.span(0)[0] if next_file_match else None]

            # Extract line numbers of the changes
            changed_lines = []
            for line_match in line_pattern.finditer(file_diff):
                start_line = int(line_match.group(1))

                # The start and end positions for this chunk of diff
                chunk_start = line_match.end()
                next_chunk = line_pattern.search(file_diff, pos=line_match.span(0)[1])
                chunk_diff = file_diff[chunk_start : next_chunk.span(0)[0] if next_chunk else None]

                lines = chunk_diff.splitlines()
                line_counter = 0
                for line in lines:
                    if line.startswith("+"):
                        changed_lines.append(start_line + line_counter)
                        line_counter += 1

            if changed_lines:
                files[file_name] = changed_lines

    return files


def get_changed_files(common_ancestor, feature_branch):
    """Get a dictionary of files and their changed lines between the common ancestor and feature_branch."""
    cmd = ["git", "diff", "-U0", "--ignore-all-space", common_ancestor, feature_branch]
    result = subprocess.check_output(cmd).decode("utf-8")

    return parse_diff_output(result)


def is_part_of_pr_changes(file_path, issue_file_line, files_changed_in_pr):
    """
    Check if a given file and line number corresponds to a change in the files included in a pull request.

    Args:
        file_path (str): The path to the file in question.
        issue_file_line (int): The line number within the file to check.
        files_changed_in_pr (dict): A dictionary of files changed in a pull request, where the keys are file paths
                                    and the values are tuples of the form (status, lines_changed_for_file), where
                                    status is a string indicating the change status ("added", "modified", or "removed"),
                                    and lines_changed_for_file is a list of tuples, where each tuple represents a range
                                    of lines changed in the file (e.g. [(10, 15), (20, 25)] indicates that lines 10-15
                                    and 20-25 were changed in the file).

    Returns:
        bool: True if the file and line number correspond to a change in the pull request, False otherwise.
    """

    if ONLY_PR_CHANGES == "false":
        return True

    debug_print(f"Looking for issue found in file={file_path} at line={issue_file_line}...")
    for file, lines_changed_for_file in files_changed_in_pr.items():
        debug_print(f'Changed file by this PR "{file}" with changed lines "{lines_changed_for_file}"')
        if file == file_path:
            for line in lines_changed_for_file:
                if line == issue_file_line:
                    debug_print(f"Issue line {issue_file_line} is a part of PR!")
                    return True

    return False


def get_lines_changed_from_patch(patch):
    """
    Parses a unified diff patch and returns the range of lines that were changed.

    Parameters:
        patch (str): The unified diff patch to parse.

    Returns:
        list: A list of tuples containing the beginning and ending line numbers for each
        section of the file that was changed by the patch.
    """

    lines_changed = []
    lines = patch.split("\n")

    for line in lines:
        # Example line @@ -43,6 +48,8 @@
        # ------------ ^
        if line.startswith("@@"):
            # Example line @@ -43,6 +48,8 @@
            # ----------------------^
            idx_beg = line.index("+")

            # Example line @@ -43,6 +48,8 @@
            #                       ^--^
            try:
                idx_end = line[idx_beg:].index(",")
                line_begin = int(line[idx_beg + 1 : idx_beg + idx_end])

                idx_beg = idx_beg + idx_end
                idx_end = line[idx_beg + 1 :].index("@@")

                num_lines = int(line[idx_beg + 1 : idx_beg + idx_end])
            except ValueError:
                # Special case for single line files
                # such as @@ -0,0 +1 @@
                idx_end = line[idx_beg:].index(" ")
                line_begin = int(line[idx_beg + 1 : idx_beg + idx_end])
                num_lines = 0

            lines_changed.append((line_begin, line_begin + num_lines))

    return lines_changed


def check_for_char_limit(incoming_line):
    return (CURRENT_COMMENT_LENGTH + len(incoming_line)) <= COMMENT_MAX_SIZE


def is_excluded_dir(line):
    """
    Determines if a given line is from a directory that should be excluded from processing.

    Args:
        line (str): The line to check.

    Returns:
        bool: True if the line is from a directory that should be excluded, False otherwise.
    """

    # In future this could be multiple different directories
    exclude_dir = os.getenv("EXCLUDE_DIR")
    if not exclude_dir:
        return False

    excluded_dir = f"{WORK_DIR}/{exclude_dir}"
    debug_print(f"{line} and {excluded_dir} with result {line.startswith(excluded_dir)}")

    return line.startswith(excluded_dir)


def get_file_line_end(file_in, file_line_start_in):
    """
    Returns the ending line number for a given file, starting from a specified line number.

    Args:
        file_in (str): The name of the file to read.
        file_line_start_in (int): The starting line number.

    Returns:
        int: The ending line number, which is either `file_line_start + 5`
        or the total number of lines in the file, whichever is smaller.
    """

    with open(f"{WORK_DIR}/{file_in}", encoding="utf-8") as file:
        num_lines = sum(1 for line in file)

    return min(file_line_start_in + 5, num_lines)


def generate_description(is_note, was_note, file_line_start, issue_description, output_string):
    """Generate description for an issue

    is_note -- is the current issue a Note: or not
    was_note -- was the previous issue a Note: or not
    file_line_start -- line to which the issue corresponds
    issue_description -- the description from cppcheck
    output_string -- entire description (can be altered if the current/previous issue is/was Note:)
    """
    global CURRENT_COMMENT_LENGTH

    if not is_note:
        description = f"\n```diff\n!Line: {file_line_start} - {issue_description}\n``` \n"
    else:
        if not was_note:
            # Previous line consists of ```diff <content> ```, so remove the closing ```
            # and append the <content> with Note: ...`

            # 12 here means "``` \n<br>\n"`
            num_chars_to_remove = 12
        else:
            # Previous line is Note: so it ends with "``` \n"
            num_chars_to_remove = 6

        output_string = output_string[:-num_chars_to_remove]
        CURRENT_COMMENT_LENGTH -= num_chars_to_remove
        description = f"\n!Line: {file_line_start} - {issue_description}``` \n"

    return output_string, description


def create_or_edit_comment(comment_body):
    """
    Creates or edits a comment on a pull request with the given comment body.

    Args:
    - comment_body: A string containing the full comment body to be created or edited.

    Returns:
    - None.
    """

    github = Github(GITHUB_TOKEN)
    repo = github.get_repo(TARGET_REPO_NAME)
    pull_request = repo.get_pull(int(PR_NUM))

    comments = pull_request.get_issue_comments()
    found_id = -1
    comment_to_edit = None
    for comment in comments:
        if (comment.user.login == "github-actions[bot]") and (COMMENT_TITLE in comment.body):
            found_id = comment.id
            comment_to_edit = comment
            break

    if found_id != -1 and comment_to_edit:
        comment_to_edit.edit(body=comment_body)
    else:
        pull_request.create_issue_comment(body=comment_body)


def generate_output(is_note, file_path, file_line_start, file_line_end, description):
    """
    Generate a formatted output string based on the details of a code issue.

    This function takes information about a code issue and constructs a string that
    includes details such as the location of the issue in the codebase, the affected code
    lines, and a description of the issue. If the issue is a note, only the description
    is returned. If the issue occurs in a different repository than the target, it
    also fetches the lines where the issue was detected.

    Parameters:
    - is_note (bool): Whether the issue is just a note or a code issue.
    - file_path (str): Path to the file where the issue was detected.
    - file_line_start (int): The line number in the file where the issue starts.
    - file_line_end (int): The line number in the file where the issue ends.
    - description (str): Description of the issue.

    Returns:
    - str: Formatted string with details of the issue.

    Note:
    - This function relies on several global variables like TARGET_REPO_NAME, REPO_NAME,
      FILES_WITH_ISSUES, and SHA which should be set before calling this function.
    """

    if not is_note:
        if TARGET_REPO_NAME != REPO_NAME:
            if file_path not in FILES_WITH_ISSUES:
                try:
                    with open(f"{file_path}", encoding="utf-8") as file:
                        lines = file.readlines()
                        FILES_WITH_ISSUES[file_path] = lines
                except FileNotFoundError:
                    print(f"Error: The file '{file_path}' was not found.")

            modified_content = FILES_WITH_ISSUES[file_path][file_line_start - 1 : file_line_end - 1]

            debug_print(f"generate_output for following file: \nfile_path={file_path} \nmodified_content={modified_content}\n")

            modified_content[0] = modified_content[0][:-1] + " <---- HERE\n"
            file_content = "".join(modified_content)

            file_url = f"https://github.com/{REPO_NAME}/blob/{SHA}/{file_path}#L{file_line_start}"
            new_line = (
                "\n\n------"
                f"\n\n <b><i>Issue found in file</b></i> [{REPO_NAME}/{file_path}]({file_url})\n"
                f"{file_content}"
                f"\n``` \n"
                f"{description} <br>\n"
            )

        else:
            new_line = (
                f"\n\nhttps://github.com/{REPO_NAME}/blob/{SHA}/{file_path}"
                f"#L{file_line_start}-L{file_line_end} {description} <br>\n"
            )
    else:
        new_line = description

    return new_line


def extract_info(line, prefix):
    """
    Extracts information from a given line containing file path, line number, and issue description.

    Args:
    - line (str): The input string containing file path, line number, and issue description.
    - prefix (str): The prefix to remove from the start of the file path in the line.
    - was_note (bool): Indicates if the previous issue was a note.
    - output_string (str): The string containing previous output information.

    Returns:
    - tuple: A tuple containing:
        - file_path (str): The path to the file.
        - is_note (bool): A flag indicating if the issue is a note.
        - description (str): Description of the issue.
        - file_line_start (int): The starting line number of the issue.
        - file_line_end (int): The ending line number of the issue.
    """

    # Clean up line
    line = line.replace(prefix, "").lstrip("/")

    # Get the line starting position /path/to/file:line and trim it
    file_path_end_idx = line.index(":")
    file_path = line[:file_path_end_idx]

    # Extract the lines information
    line = line[file_path_end_idx + 1 :]

    # Get line (start, end)
    file_line_start = int(line[: line.index(":")])
    file_line_end = get_file_line_end(file_path, file_line_start)

    # Get content of the issue
    issue_description = line[line.index(" ") + 1 :]
    is_note = issue_description.startswith("note:")

    return (file_path, is_note, file_line_start, file_line_end, issue_description)


def create_common_input_vars_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_to_console",
        help="Whether to output the result to console",
        required=True,
    )
    parser.add_argument(
        "-fk",
        "--fork_repository",
        help="Whether the actual code is in 'pr_tree' directory",
        required=True,
    )
    parser.add_argument(
        "--common",
        default="",
        help="common ancestor between two branches (default: %(default)s)",
    )
    parser.add_argument("--head", default="", help="Head branch (default: %(default)s)")

    return parser


def append_issue(is_note, per_issue_string, new_line, list_of_issues):
    if not is_note:
        if len(per_issue_string) > 0 and (per_issue_string not in list_of_issues):
            list_of_issues.append(per_issue_string)
        per_issue_string = new_line
    else:
        per_issue_string += new_line

    return per_issue_string


def create_comment_for_output(tool_output, prefix, files_changed_in_pr, output_to_console):
    """
    Generates a comment for a GitHub pull request based on the tool output.

    Parameters:
        tool_output (str): The tool output to parse.
        prefix (str): The prefix to look for in order to identify issues.
        files_changed_in_pr (dict): A dictionary containing the files that were
            changed in the pull request and the lines that were modified.
        output_to_console (bool): Whether or not to output the results to the console.

    Returns:
        tuple: A tuple containing the generated comment and the number of issues found.
    """
    list_of_issues = []
    per_issue_string = ""
    was_note = False

    for line in tool_output:
        if line.startswith(prefix) and not is_excluded_dir(line):
            (
                file_path,
                is_note,
                file_line_start,
                file_line_end,
                issue_description,
            ) = extract_info(line, prefix)

            # In case where we only output to console, skip the next part
            if output_to_console:
                per_issue_string = append_issue(is_note, per_issue_string, line, list_of_issues)
                continue

            if is_part_of_pr_changes(file_path, file_line_start, files_changed_in_pr):
                per_issue_string, description = generate_description(
                    is_note,
                    was_note,
                    file_line_start,
                    issue_description,
                    per_issue_string,
                )
                was_note = is_note
                new_line = generate_output(is_note, file_path, file_line_start, file_line_end, description)

                global CURRENT_COMMENT_LENGTH
                if check_for_char_limit(new_line):
                    per_issue_string = append_issue(is_note, per_issue_string, new_line, list_of_issues)
                    CURRENT_COMMENT_LENGTH += len(new_line)

                else:
                    CURRENT_COMMENT_LENGTH = COMMENT_MAX_SIZE

                    return "\n".join(list_of_issues), len(list_of_issues)

    # Append any unprocessed issues
    if len(per_issue_string) > 0 and (per_issue_string not in list_of_issues):
        list_of_issues.append(per_issue_string)

    output_string = "\n".join(list_of_issues)

    debug_print(f"\nFinal output_string = \n{output_string}\n")

    return output_string, len(list_of_issues)


def read_files_and_parse_results():
    """Reads the output files generated by cppcheck and creates comments
    for the pull request, based on the issues found. The comments can be output to console
    and/or added to the pull request. Returns a tuple with the comments generated for
    cppcheck, and boolean values indicating whether issues were found by each tool,
    whether output was generated to the console, and whether the actual code
    is in the 'pr_tree' directory.

    Returns:
        A tuple with the following values:
        - cppcheck_comment (str): The comment generated for cppcheck, if any issues were found.
        - cppcheck_issues_found (bool): Whether issues were found by cppcheck.
        - output_to_console (bool): Whether output was generated to the console.
    """

    # Get cppcheck files
    parser = create_common_input_vars_parser()
    parser.add_argument("-cc", "--cppcheck", help="Output file name for cppcheck", required=True)

    if parser.parse_args().fork_repository == "true":
        # Make sure to use Head repository
        global REPO_NAME
        REPO_NAME = os.getenv("PR_REPO")

    cppcheck_file_name = parser.parse_args().cppcheck
    output_to_console = parser.parse_args().output_to_console == "true"

    cppcheck_content = ""
    with open(cppcheck_file_name, "r", encoding="utf-8") as file:
        cppcheck_content = file.readlines()

    common_ancestor = parser.parse_args().common
    feature_branch = parser.parse_args().head

    line_prefix = f"{WORK_DIR}"

    debug_print(f"cppcheck result: \n {cppcheck_content} \n" f"line_prefix: {line_prefix} \n")

    files_changed_in_pr = {}
    if not output_to_console and (ONLY_PR_CHANGES == "true"):
        files_changed_in_pr = get_changed_files(common_ancestor, feature_branch)

    cppcheck_comment, cppcheck_issues_found = create_comment_for_output(
        cppcheck_content, line_prefix, files_changed_in_pr, output_to_console
    )

    if output_to_console and cppcheck_issues_found:
        print("##[error] Issues found!\n")
        error_color = "\u001b[31m"

        if cppcheck_issues_found:
            print(f"{error_color}cppcheck results: {cppcheck_comment}")

    return cppcheck_comment, cppcheck_issues_found, output_to_console


def prepare_comment_body(cppcheck_comment, cppcheck_issues_found):
    """
    Generates a comment body based on the results of the cppcheck analysis.

    Args:
        cppcheck_comment (str): The comment body generated for the cppcheck analysis.
        cppcheck_issues_found (int): The number of issues found by cppcheck analysis.

    Returns:
        str: The final comment body that will be posted as a comment on the pull request.
    """

    if cppcheck_issues_found == 0:
        full_comment_body = (
            '## <p align="center"><b> :white_check_mark:' f"{COMMENT_TITLE} - no issues found! :white_check_mark: </b></p>"
        )
    else:
        full_comment_body = f'## <p align="center"><b> :zap: {COMMENT_TITLE} :zap: </b></p> \n\n'

        if len(cppcheck_comment) > 0:
            full_comment_body += (
                f"<details> <summary> <b> :red_circle: cppcheck found "
                f"{cppcheck_issues_found} {'issues' if cppcheck_issues_found > 1 else 'issue'}!"
                " Click here to see details. </b> </summary> <br>"
                f"{cppcheck_comment} </details>"
            )

    if CURRENT_COMMENT_LENGTH == COMMENT_MAX_SIZE:
        full_comment_body += f"\n```diff\n{MAX_CHAR_COUNT_REACHED}\n```"

    debug_print(f"Repo={REPO_NAME} pr_num={PR_NUM} comment_title={COMMENT_TITLE}")

    return full_comment_body


if __name__ == "__main__":
    cppcheck_comment_in, cppcheck_issues_found_in, output_to_console_in = read_files_and_parse_results()

    if not output_to_console_in:
        comment_body_in = prepare_comment_body(cppcheck_comment_in, cppcheck_issues_found_in)
        create_or_edit_comment(comment_body_in)

    sys.exit(cppcheck_issues_found_in)
