#
# This source file is part of the EdgeDB open source project.
#
# Copyright 2017-present MagicStack Inc. and the EdgeDB authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from __future__ import annotations
from typing import Any, TextIO
from typing_extensions import TypeAliasType
from collections.abc import Callable

import sys

import click


Style = TypeAliasType("Style", Callable[[str], str])


def marker_passed(t: str) -> str:
    return t


def marker_errored(t: str) -> str:
    return click.style(t, fg="red", bold=True)


def marker_skipped(t: str) -> str:
    return click.style(t, fg="yellow")


def marker_failed(t: str) -> str:
    return click.style(t, fg="red", bold=True)


def marker_xfailed(t: str) -> str:
    return t


def marker_not_implemented(t: str) -> str:
    return t


def marker_upassed(t: str) -> str:
    return click.style(t, fg="yellow")


def status(t: str) -> str:
    return click.style(t, fg="white", bold=True)


def warning(t: str) -> str:
    return click.style(t, fg="yellow")


class ClickUI:
    def __init__(
        self, verbosity: int = 1, stream: TextIO = sys.stderr
    ) -> None:
        self._verbosity = verbosity
        self._stream = stream

    def _echo(self, msg: str = "", **kwargs: Any) -> None:
        if self._verbosity > 0:
            click.secho(msg, file=self._stream, **({"nl": False} | kwargs))

    def text(self, msg: str) -> None:
        if self._verbosity > 1:
            self._echo(msg)

    def info(self, msg: str) -> None:
        self._echo(msg, fg="white")

    def warning(self, msg: str) -> None:
        self._echo(msg, fg="yellow")

    def error(self, msg: str) -> None:
        self._echo(msg, fg="read")
