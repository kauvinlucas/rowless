"""Test package for rowless-dbt.

A package rather than a loose directory, so that this suite can have its own
`conftest.py` without colliding with the one in rowless-core: two files called
`conftest` in one workspace are one module name to mypy, and test files named
alike are one module to pytest. The `__init__.py` makes every module here
`tests.*` instead.
"""
