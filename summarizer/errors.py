"""The one error type the CLI knows how to show nicely."""


class SummarizerError(Exception):
    """Something went wrong, and the message says how to fix it.

    The CLI catches this, prints the message, and exits with code 1 instead of
    dumping a Python traceback on the user.
    """
