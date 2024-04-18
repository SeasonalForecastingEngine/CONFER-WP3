# Running tests in Python using Pytest


Python files starting with `test_*` are searched by `pytest` for functions starting with `test_` as well. These functions are then run, and
any `assert (statement) "message"` where `statement == True` signify that a test passed, whilst the test failed if `statement == False`.

You can run the tests yourself by running the command
- `poetry run pytest`

in your terminal, and they will also be run automatically when you make a pull request or push changes to the `main` or `master` branch.
