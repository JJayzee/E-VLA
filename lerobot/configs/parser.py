# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import inspect
import pkgutil
import sys
from argparse import ArgumentError
from functools import wraps
from pathlib import Path
from typing import Sequence

import draccus

from lerobot.utils.utils import has_method

PATH_KEY = "path"
PLUGIN_DISCOVERY_SUFFIX = "discover_packages_path"
draccus.set_config_type("json")


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def resolve_event_window_args(
    field_names: str | Sequence[str],
    args: Sequence[str] | None = None,
    *,
    required: bool = True,
) -> tuple[int | None, int | None]:
    """Resolve one training event-window mode from one or more CLI namespaces.

    A window may be repeated in multiple namespaces (for example ``dataset`` and
    ``policy``) for backwards compatibility, but all repeated values must agree.
    The checkpoint configuration is deliberately not consulted here; this helper
    is only used before an event-training policy is loaded.
    """
    if args is None:
        args = sys.argv[1:]
    if isinstance(field_names, str):
        field_names = [field_names]

    values: dict[str, list[str]] = {"event_window_size": [], "event_window_duration": []}
    for field_name in field_names:
        for window_name in values:
            prefix = f"--{field_name}.{window_name}="
            values[window_name].extend(arg[len(prefix) :] for arg in args if arg.startswith(prefix))

    provided_modes = [name for name, raw_values in values.items() if raw_values]
    field_hint = " or ".join(
        f"--{field_name}.event_window_size / --{field_name}.event_window_duration"
        for field_name in field_names
    )
    if not provided_modes:
        if required:
            raise ValueError(
                f"Exactly one event window must be provided for event training ({field_hint}); neither was provided."
            )
        return None, None
    if len(provided_modes) != 1:
        raise ValueError(
            "event_window_size and event_window_duration are mutually exclusive; "
            "provide exactly one for event training."
        )

    selected_mode = provided_modes[0]
    try:
        parsed_values = [int(value) for value in values[selected_mode]]
    except ValueError as exc:
        raise ValueError(f"{selected_mode} must be a positive integer.") from exc
    if any(value <= 0 for value in parsed_values):
        unit = " duration in microseconds" if selected_mode == "event_window_duration" else " integer"
        raise ValueError(f"{selected_mode} must be a positive{unit}.")
    if len(set(parsed_values)) != 1:
        raise ValueError(f"Conflicting runtime values were provided for {selected_mode}: {parsed_values}.")

    value = parsed_values[0]
    if selected_mode == "event_window_size":
        return value, None
    return None, value


def set_event_window_cli_overrides(
    cli_overrides: Sequence[str],
    event_window_size: int | None,
    event_window_duration: int | None,
) -> list[str]:
    """Replace checkpoint window fields with the selected training window."""
    if (event_window_size is None) == (event_window_duration is None):
        raise ValueError("Exactly one training event window must be selected before applying policy overrides.")

    filtered = [
        arg
        for arg in cli_overrides
        if not arg.startswith("--event_window_size=") and not arg.startswith("--event_window_duration=")
    ]
    filtered.extend(
        [
            f"--event_window_size={event_window_size if event_window_size is not None else 'null'}",
            f"--event_window_duration={event_window_duration if event_window_duration is not None else 'null'}",
        ]
    )
    return filtered


def strip_event_window_cli_overrides(cli_overrides: Sequence[str]) -> list[str]:
    """Drop event-window CLI overrides when loading a policy for inference."""
    return [
        arg
        for arg in cli_overrides
        if not arg.startswith("--event_window_size=") and not arg.startswith("--event_window_duration=")
    ]


def parse_plugin_args(plugin_arg_suffix: str, args: Sequence[str]) -> dict:
    """Parse plugin-related arguments from command-line arguments.

    This function extracts arguments from command-line arguments that match a specified suffix pattern.
    It processes arguments in the format '--key=value' and returns them as a dictionary.

    Args:
        plugin_arg_suffix (str): The suffix to identify plugin-related arguments.
        cli_args (Sequence[str]): A sequence of command-line arguments to parse.

    Returns:
        dict: A dictionary containing the parsed plugin arguments where:
            - Keys are the argument names (with '--' prefix removed if present)
            - Values are the corresponding argument values

    Example:
        >>> args = ['--plugin.discover_packages_path=my_package',
        ...         '--other_arg=value']
        >>> parse_plugin_args('discover_packages_path', args)
        {'plugin.discover_packages_path': 'my_package'}
    """
    plugin_args = {}
    for arg in args:
        if "=" in arg and plugin_arg_suffix in arg:
            key, value = arg.split("=", 1)
            # Remove leading '--' if present
            if key.startswith("--"):
                key = key[2:]
            plugin_args[key] = value
    return plugin_args


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""


def load_plugin(plugin_path: str) -> None:
    """Load and initialize a plugin from a given Python package path.

    This function attempts to load a plugin by importing its package and any submodules.
    Plugin registration is expected to happen during package initialization, for example
    when config classes are registered with their parents using the `register_subclass` decorator.

    Args:
        plugin_path (str): The Python package path to the plugin (e.g. "mypackage.plugins.myplugin")

    Raises:
        PluginLoadError: If the plugin cannot be loaded due to import errors or if the package path is invalid.

    Examples:
        >>> load_plugin("external_plugin.core")       # Loads plugin from external package

    Notes:
        - The plugin package should handle its own registration during import
        - All submodules in the plugin package will be imported
        - Implementation follows the plugin discovery pattern from Python packaging guidelines

    See Also:
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """
    try:
        package_module = importlib.import_module(plugin_path, __package__)
    except (ImportError, ModuleNotFoundError) as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e

    def iter_namespace(ns_pkg):
        return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

    try:
        for _finder, pkg_name, _ispkg in iter_namespace(package_module):
            importlib.import_module(pkg_name)
    except ImportError as e:
        raise PluginLoadError(
            f"Failed to load plugin '{plugin_path}'. Verify the path and installation: {str(e)}"
        ) from e


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def filter_arg(field_to_filter: str, args: Sequence[str] | None = None) -> list[str]:
    return [arg for arg in args if not arg.startswith(f"--{field_to_filter}=")]


def filter_path_args(fields_to_filter: str | list[str], args: Sequence[str] | None = None) -> list[str]:
    """
    Filters command-line arguments related to fields with specific path arguments.

    Args:
        fields_to_filter (str | list[str]): A single str or a list of str whose arguments need to be filtered.
        args (Sequence[str] | None): The sequence of command-line arguments to be filtered.
            Defaults to None.

    Returns:
        list[str]: A filtered list of arguments, with arguments related to the specified
        fields removed.

    Raises:
        ArgumentError: If both a path argument (e.g., `--field_name.path`) and a type
            argument (e.g., `--field_name.type`) are specified for the same field.
    """
    if isinstance(fields_to_filter, str):
        fields_to_filter = [fields_to_filter]

    filtered_args = args
    for field in fields_to_filter:
        if get_path_arg(field, args):
            if get_type_arg(field, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field}.{PATH_KEY} and --{field}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field}.")]

    return filtered_args


def wrap(config_path: Path | None = None):
    """
    HACK: Similar to draccus.wrap but does three additional things:
        - Will remove '.path' arguments from CLI in order to process them later on.
        - If a 'config_path' is passed and the main config class has a 'from_pretrained' method, will
          initialize it from there to allow to fetch configs from the hub directly
        - Will load plugins specified in the CLI arguments. These plugins will typically register
            their own subclasses of config classes, so that draccus can find the right class to instantiate
            from the CLI '.type' arguments
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                plugin_args = parse_plugin_args(PLUGIN_DISCOVERY_SUFFIX, cli_args)
                for plugin_cli_arg, plugin_path in plugin_args.items():
                    try:
                        load_plugin(plugin_path)
                    except PluginLoadError as e:
                        # add the relevant CLI arg to the error message
                        raise PluginLoadError(f"{e}\nFailed plugin CLI Arg: {plugin_cli_arg}") from e
                    cli_args = filter_arg(plugin_cli_arg, cli_args)
                config_path_cli = parse_arg("config_path", cli_args)
                if has_method(argtype, "__get_path_fields__"):
                    path_fields = argtype.__get_path_fields__()
                    cli_args = filter_path_args(path_fields, cli_args)
                if has_method(argtype, "from_pretrained") and config_path_cli:
                    cli_args = filter_arg("config_path", cli_args)
                    cfg = argtype.from_pretrained(config_path_cli, cli_args=cli_args)
                else:
                    cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
