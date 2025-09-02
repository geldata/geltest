#
# This source file is part of the EdgeDB open source project.
#
# Copyright 2016-present MagicStack Inc. and the EdgeDB authors.
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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from typing_extensions import TypeAliasType
from collections.abc import Mapping

import contextlib
import heapq
import importlib
import importlib.machinery
import importlib.util
import pathlib
import pickle  # noqa: S403
import re
import sys
import unittest


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence


Client = TypeAliasType("Client", Any)


class UI(Protocol):
    def text(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None: ...


@runtime_checkable
class Fixture(Protocol):
    def __get__(
        self,
        instance: Any | None,
        owner: type[Any] | None = None,
        /,
    ) -> Any: ...
    def set_options(self, options: Mapping[str, str]) -> None: ...
    def get_shared_data(self) -> object: ...
    def set_shared_data(self, data: object) -> None: ...
    async def set_up(self, ui: UI) -> None: ...
    async def tear_down(self, ui: UI) -> None: ...
    async def post_session_set_up(
        self, cases: Sequence[type[Any]], *, ui: UI
    ) -> None: ...


@runtime_checkable
class DatabaseTestCaseProto(Protocol):
    @classmethod
    def set_options(cls, options: Mapping[str, str]) -> None: ...

    @classmethod
    async def set_up_class_once(cls, ui: UI) -> None: ...

    @classmethod
    async def tear_down_class_once(cls, ui: UI) -> None: ...

    @classmethod
    def get_shared_data(cls) -> Mapping[str, object]: ...

    @classmethod
    def update_shared_data(cls, **data: object) -> None: ...


class TestLoader(unittest.TestLoader):
    include: list[re.Pattern[str]] | None
    exclude: list[re.Pattern[str]] | None

    def __init__(
        self,
        *,
        verbosity: int = 1,
        exclude: Sequence[str] = (),
        include: Sequence[str] = (),
        progress_cb: Callable[[int, int], None] | None = None,
    ):
        super().__init__()
        self.verbosity = verbosity

        if include:
            self.include = [re.compile(r) for r in include]
        else:
            self.include = None

        if exclude:
            self.exclude = [re.compile(r) for r in exclude]
        else:
            self.exclude = None

        self.progress_cb = progress_cb

    def getTestCaseNames(
        self, testCaseClass: type[unittest.TestCase]
    ) -> Sequence[str]:
        names = super().getTestCaseNames(testCaseClass)
        unfiltered_len = len(names)
        cname = testCaseClass.__name__

        if self.include:
            names = [
                n
                for n in names
                if (
                    any(r.search(n) for r in self.include)
                    or any(r.search(f"{cname}.{n}") for r in self.include)
                )
            ]

        if self.exclude:
            names = [
                n
                for n in names
                if (
                    not any(r.search(n) for r in self.exclude)
                    and not any(r.search(f"{cname}.{n}") for r in self.exclude)
                )
            ]

        if self.progress_cb:
            self.progress_cb(len(names), unfiltered_len)

        return names


def _add_test(
    result: dict[
        type[unittest.TestCase],
        tuple[list[unittest.TestCase], list[unittest.TestCase]],
    ],
    test: tuple[unittest.TestCase, ...],
) -> None:
    # test is a tuple of the same test method that may zREPEAT
    cls = type(test[0])
    try:
        methods, repeat_methods = result[cls]
    except KeyError:
        # put zREPEAT tests in a separate list
        methods = []
        repeat_methods = []
        result[cls] = methods, repeat_methods

    methods.append(test[0])
    if len(test) > 1:
        repeat_methods.extend(test[1:])


def _merge_results(
    result: dict[
        type[unittest.TestCase],
        tuple[list[unittest.TestCase], list[unittest.TestCase]],
    ],
) -> Mapping[type[unittest.TestCase], Sequence[unittest.TestCase]]:
    # make sure all the zREPEAT tests comes in the end
    return {k: v[0] + v[1] for k, v in result.items()}


def _get_test_cases(
    tests: Iterable[unittest.TestCase | unittest.TestSuite],
) -> dict[
    type[unittest.TestCase],
    tuple[list[unittest.TestCase], list[unittest.TestCase]],
]:
    result: dict[
        type[unittest.TestCase],
        tuple[list[unittest.TestCase], list[unittest.TestCase]],
    ] = {}

    for test in tests:
        if isinstance(test, unittest.TestSuite):
            result.update(_get_test_cases(test))
        elif not getattr(test, "__unittest_skip__", False):
            _add_test(result, (test,))

    return result


def get_test_cases(
    tests: Iterable[unittest.TestCase | unittest.TestSuite],
) -> Mapping[type[unittest.TestCase], Sequence[unittest.TestCase]]:
    return _merge_results(_get_test_cases(tests))


def get_cases_by_shard(
    cases: Mapping[type[unittest.TestCase], Sequence[unittest.TestCase]],
    selected_shard: int,
    total_shards: int,
    verbosity: int,
    stats: dict[str, tuple[float, int]],
) -> Mapping[type[unittest.TestCase], Sequence[unittest.TestCase]]:
    if total_shards <= 1:
        return cases

    selected_shard -= 1  # starting from 0
    new_test_est = 0.1  # default estimate if test is not found in stats
    new_setup_est = 1  # default estimate if setup is not found in stats

    # For logging
    total_tests = 0
    selected_tests = 0
    total_est = 0.0
    selected_est = 0.0

    # Priority queue of tests grouped by setup script ordered by estimated
    # running time of the groups. Order of tests within cases is preserved.
    tests_by_setup: list[
        tuple[
            float,
            int,
            float,
            list[tuple[float, tuple[unittest.TestCase, ...]]],
        ]
    ] = []

    # Priority queue of individual tests ordered by estimated running time.
    tests_with_est: list[tuple[float, int, tuple[unittest.TestCase, ...]]] = []

    # Prepare the source heaps
    setup_count = 0
    for case, tests in cases.items():
        # Extract zREPEAT tests and attach them to their first runs
        combined: dict[str, tuple[unittest.TestCase, ...]] = {}
        for test in tests:
            test_name = str(test)
            orig_name = test_name.replace("test_zREPEAT", "test")
            if orig_name == test_name:
                if test_name in combined:
                    combined[test_name] = (test, *combined[test_name])
                else:
                    combined[test_name] = (test,)
            else:
                if orig_name in combined:
                    combined[orig_name] = (*combined[orig_name], test)
                else:
                    combined[orig_name] = (test,)

        setup_script_getter = getattr(case, "get_setup_script", None)
        if setup_script_getter and combined:
            tests_per_setup = []
            database_name_getter = getattr(case, "get_database_name", None)
            database_name = (
                database_name_getter() if database_name_getter else "unknown"
            )
            est_per_setup = setup_est = stats.get(
                "setup::" + database_name,
                (new_setup_est, 0),
            )[0]
            for test_name, test_group in combined.items():
                total_tests += len(test_group)
                est = stats.get(test_name, (new_test_est, 0))[0] * len(
                    test_group
                )
                est_per_setup += est
                tests_per_setup.append((est, test_group))
            heapq.heappush(
                tests_by_setup,
                (-est_per_setup, setup_count, setup_est, tests_per_setup),
            )
            setup_count += 1
            total_est += est_per_setup
        else:
            for test_name, test_group in combined.items():
                total_tests += len(test_group)
                est = stats.get(test_name, (new_test_est, 0))[0] * len(
                    test_group
                )
                total_est += est
                heapq.heappush(tests_with_est, (-est, total_tests, test_group))

    target_est = total_est / total_shards  # target running time of one shard
    shards_est: list[tuple[float, int, set[int]]] = [
        (0.0, shard, set()) for shard in range(total_shards)
    ]
    result_cases: dict[
        type[unittest.TestCase],
        tuple[list[unittest.TestCase], list[unittest.TestCase]],
    ] = {}  # output
    setup_to_alloc = set(range(setup_count))  # tracks first run of each setup

    # Assign per-setup tests first
    while tests_by_setup:
        remaining_est, setup_id, setup_est, tests_per_setup = heapq.heappop(
            tests_by_setup,
        )
        est_acc, current, setups = heapq.heappop(shards_est)

        # Add setup time
        if setup_id not in setups:
            setups.add(setup_id)
            est_acc += setup_est
            if current == selected_shard:
                selected_est += setup_est
            if setup_id in setup_to_alloc:
                setup_to_alloc.remove(setup_id)
            else:
                # This means one more setup for the overall test run
                target_est += setup_est / total_shards

        # Add as much tests from this group to current shard as possible
        while tests_per_setup:
            est, tests = tests_per_setup.pop(0)
            est_acc += est  # est is a positive number
            remaining_est += est  # remaining_est is a negative number

            if current == selected_shard:
                # Add the test to the result
                _add_test(result_cases, tests)
                selected_tests += len(tests)
                selected_est += est

            if est_acc >= target_est and -remaining_est > setup_est * 2:
                # Current shard is full and the remaining tests would take more
                # time than their setup, then add the tests back to the heap so
                # that we could add them to another shard
                heapq.heappush(
                    tests_by_setup,
                    (remaining_est, setup_id, setup_est, tests_per_setup),
                )
                break

        heapq.heappush(shards_est, (est_acc, current, setups))

    # Assign all non-setup tests, but leave the last shard for everything else
    setups = set()
    while tests_with_est and len(shards_est) > 1:
        est, _, tests = heapq.heappop(tests_with_est)  # est is negative
        est_acc, current, setups = heapq.heappop(shards_est)
        est_acc -= est

        if current == selected_shard:
            # Add the test to the result
            _add_test(result_cases, tests)
            selected_tests += len(tests)
            selected_est -= est

        if est_acc >= target_est:
            # The current shard is full
            if current == selected_shard:
                # End early if the selected shard is full
                break
        else:
            # Only add the current shard back to the heap if it's not full
            heapq.heappush(shards_est, (est_acc, current, setups))

    else:
        # Add all the remaining tests to the first remaining shard if any
        while shards_est:
            est_acc, current, setups = heapq.heappop(shards_est)
            if current == selected_shard:
                for est, _, tests in tests_with_est:
                    _add_test(result_cases, tests)
                    selected_tests += len(tests)
                    selected_est -= est
                break
            tests_with_est.clear()  # should always be empty already here

    if verbosity >= 1:
        pass
    return _merge_results(result_cases)


def _set_sys_path(entries: list[str]) -> None:
    sys.path[:] = entries
    importlib.invalidate_caches()


@contextlib.contextmanager
def _sys_path(*paths: str) -> Iterator[None]:
    """Modify sys.path by temporarily placing the given entry in front"""
    orig_sys_path = sys.path[:]
    paths_set = {*paths}
    _set_sys_path([*paths, *(p for p in orig_sys_path if p not in paths_set)])
    try:
        yield
    finally:
        _set_sys_path(orig_sys_path)


_ImportLocation = TypeAliasType("_ImportLocation", tuple[list[str], str, str])


def _get_module_spec(modname: str) -> importlib.machinery.ModuleSpec:
    spec: importlib.machinery.ModuleSpec | None = None
    if (module := sys.modules.get(modname)) is not None:
        spec = module.__spec__

    if spec is None:
        spec = importlib.util.find_spec(modname)

    if spec is None:
        raise RuntimeError(f"cannot find module spec for {modname}")

    return spec


def _get_class_import_location(cls: type[Any]) -> _ImportLocation:
    top_mod, _, _ = cls.__module__.partition(".")
    top_level = _get_module_spec(top_mod)
    if not top_level.has_location:
        raise RuntimeError(
            f"{cls} originiates from a module that has no loadable location"
        )

    search_paths: list[str] = []

    if top_level.origin is not None:
        path = pathlib.Path(top_level.origin)
        if not path.exists():
            raise RuntimeError(
                f"{cls} originiates from a module that cannot be loaded from "
                f"a path: {path}"
            )

        if top_level.parent == top_level.name:
            parent_dir = path.parent.parent
        else:
            parent_dir = path.parent

        search_paths.append(str(parent_dir))

    elif locs := top_level.submodule_search_locations:
        for loc in locs:
            path = pathlib.Path(loc)
            if path.exists():
                search_paths.append(str(path.parent))

    if not search_paths:
        raise RuntimeError(
            f"{cls} originiates from a module that cannot be loaded from "
            f"any path"
        )

    return search_paths, cls.__module__, cls.__qualname__


def _restore_TestCase(
    clsloc: _ImportLocation,
    newargs: tuple[tuple[Any, ...], dict[str, Any]],
    state: object,
) -> unittest.TestCase:
    search_paths, modname, clsname = clsloc
    with _sys_path(*search_paths):
        mod = importlib.import_module(modname)

    cls = getattr(mod, clsname)
    if not issubclass(cls, unittest.TestCase):
        raise RuntimeError(f"unexpected non-TestCase type: {cls}")

    test: unittest.TestCase = cls.__new__(cls, *newargs[0], **newargs[1])
    if callable(setstate := getattr(test, "__setstate__", None)):
        setstate(state)
    elif isinstance(state, dict):
        test.__dict__.update(state)

    return test


def _reduce_TestCase(
    test: unittest.TestCase,
) -> tuple[
    Callable[..., unittest.TestCase],
    tuple[
        _ImportLocation,
        tuple[tuple[Any, ...], dict[str, Any]],
        object | None,
    ],
]:
    clsloc = _get_class_import_location(type(test))
    newargs: tuple[Any, ...]
    newkwargs: dict[str, Any]
    if (getnewargs_ex := getattr(test, "__getnewargs_ex__", None)) is not None:
        newargs, newkwargs = getnewargs_ex()
    elif (getnewargs := getattr(test, "__getnewargs__", None)) is not None:
        newargs = getnewargs()
        newkwargs = {}
    else:
        newargs = ()
        newkwargs = {}

    state: object | None
    if (getstate := getattr(test, "__getstate__", None)) is not None:
        state = getstate()
    elif (dct := getattr(test, "__dict__", None)) is not None:
        state = {**dct}
    else:
        state = None

    if isinstance(state, Mapping):
        # Check state contents for pickleability
        # and exclude attributes that are unpickleable.
        # Technically this is unsound, but the runner
        # infrastructure should not depend on any of those.
        state = {**state}
        for k, v in [*state.items()]:
            try:
                pickle.dumps(v)
            except Exception:
                state.pop(k)

    return _restore_TestCase, (clsloc, (newargs, newkwargs), state)


class TestCasePickleWrapper:
    """Makes TestCases more reliably unpickleable"""

    def __init__(self, case: unittest.TestCase) -> None:
        self._case = case

    def __reduce__(self) -> tuple[Any, ...]:
        return _reduce_TestCase(self._case)


def discover(
    paths: Sequence[str],
    *,
    verbosity: int = 1,
    exclude: Sequence[str] = (),
    include: Sequence[str] = (),
    progress_cb: Callable[[int, int], None] | None = None,
) -> unittest.TestSuite:
    test_loader = TestLoader(
        verbosity=verbosity,
        exclude=exclude,
        include=include,
        progress_cb=progress_cb,
    )

    suite = unittest.TestSuite()

    for entry in paths:
        file = pathlib.Path(entry).absolute()

        # Establish the top_level_dir as the parent of
        # the topmost directory containing __init__.py.
        # This ensures correct test module package structure
        # and enables relative imports within.
        top_level_dir = file.parent
        while (top_level_dir / "__init__.py").exists():
            top_level_dir = top_level_dir.parent

        top_level_dir_str = str(top_level_dir)
        # Make sure we import from the correct place
        with _sys_path(top_level_dir_str):
            if file.is_dir():
                tests = test_loader.discover(
                    start_dir=str(file),
                    top_level_dir=top_level_dir_str,
                )
            else:
                tests = test_loader.discover(
                    start_dir=str(file.parent),
                    pattern=file.name,
                    top_level_dir=top_level_dir_str,
                )

        suite.addTest(tests)

    return suite
