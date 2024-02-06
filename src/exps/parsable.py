from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import _MISSING_TYPE, Field, dataclass
from functools import cache, partial
from itertools import groupby
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    get_type_hints,
)
from typing_extensions import NotRequired, Self, TypedDict

ArgumentParser: TypeAlias = argparse.ArgumentParser
ActionSubParsers: TypeAlias = argparse._SubParsersAction

try:
    from rich_argparse import MetavarTypeRichHelpFormatter, RichHelpFormatter

    HelpFormatter = RichHelpFormatter
    TypeHelpFormatter = MetavarTypeRichHelpFormatter
except ImportError:
    HelpFormatter = argparse.MetavarTypeHelpFormatter
    TypeHelpFormatter = argparse.MetavarTypeHelpFormatter


class IsDataclass(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Mapping[str, Field]]


@cache
def _field_types(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


@cache
def _dataclass_fields(cls: type[IsDataclass]) -> dict[str, Field]:
    def exclude(_: str, f: Field) -> bool:
        return "ClassVar" in str(f.type) or f.init is False

    return {k: v for k, v in cls.__dataclass_fields__.items() if not exclude(k, v)}


def _bool_parser(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    if x.lower() in ("true", "1", "t", "y", "yes"):
        return True
    if x.lower() in ("false", "0", "f", "n", "no"):
        return False
    raise ValueError(f"Cannot interpret '{x}' as a boolean")


D = TypeVar("D", int, str, float, bool)
E = TypeVar("E")


class Arg(TypedDict, Generic[E, D], total=False):
    help: NotRequired[str]
    choices: NotRequired[Iterable[str]]
    group: NotRequired[str]
    encode: NotRequired[Callable[[E], D]]
    decode: NotRequired[Callable[[D], E]]
    factory: NotRequired[Callable[[dict[str, Any]], E]]
    extra: NotRequired[dict[Any, Any]]


@dataclass(kw_only=True)
class Parsable(IsDataclass):
    @classmethod
    def as_argument(cls, field: Field, parser: argparse.ArgumentParser) -> None:
        _types = _field_types(cls)
        _type = _types.get(field.name)
        if _type is None:
            raise ValueError(f"Could not find type for field {field.name} in {cls}")

        args: tuple = ()
        kwargs = {}
        _type = _bool_parser if _type is bool else _type
        kwargs["type"] = _type
        metadata = Arg(**field.metadata)
        if (_help := metadata.get("help")) is not None:
            kwargs["help"] = _help
        if (_choices := metadata.get("choices")) is not None:
            kwargs["choices"] = _choices

        if isinstance(field.default, _MISSING_TYPE):
            args = (f"--{field.name}",)
        else:
            args = (f"--{field.name}",)
            kwargs["required"] = False
            kwargs["default"] = field.default

            _help = metadata.get("help", "")
            kwargs["help"] = _help + f"\nDefault: {field.default}"

        parser.add_argument(*args, **kwargs)

    @classmethod
    def parser(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[ArgumentParser, Callable[[str], AbstractContextManager[ArgumentParser]]]:
        parser = argparse.ArgumentParser(*args, formatter_class=HelpFormatter, **kwargs)
        subparsers = parser.add_subparsers(dest="command")
        _cmd_generator = partial(cmd_generator, subparsers=subparsers)
        return parser, _cmd_generator

    @classmethod
    def parse(cls, args: argparse.Namespace) -> Self:
        _args = {k: v for k, v in vars(args).items() if k != "command"}
        for field in cls.dataclass_fields().values():
            if (_factory := field.metadata.get("factory")) is not None:
                _args[field.name] = _factory(_args)

        return cls(**_args)

    @classmethod
    def add_creation_arugments(cls, parser: argparse.ArgumentParser) -> None:
        group_fields = {}
        get_group = lambda f: f.metadata.get("group", "")
        fields = sorted(cls.dataclass_fields().values(), key=get_group)

        for group, group_fields in groupby(fields, get_group):
            if group == "":
                for field in group_fields:
                    cls.as_argument(field, parser)
            else:
                group_parser = parser.add_argument_group(group)
                for field in group_fields:
                    cls.as_argument(field, group_parser)  # type: ignore

    @classmethod
    def dataclass_fields(cls) -> dict[str, Field]:
        return _dataclass_fields(cls)

    def item_fields(self) -> dict[str, tuple[Any, Field]]:
        def exclude(_: str, f: Field) -> bool:
            return "ClassVar" in str(f.type) or f.init is False

        return {
            k: (getattr(self, k), f)
            for k, f in self.dataclass_fields().items()
            if not exclude(k, f)
        }

    def _values(self) -> dict[str, Any]:
        return {k: v for k, (v, _) in self.item_fields().items()}

    def grouped_fields(
        self,
        *,
        order: Iterable[str] | None = None,
    ) -> dict[str, list[tuple[str, Any, Field]]]:
        d: dict[str, list[tuple[str, Any, Field]]] = defaultdict(list)
        for k, (v, f) in self.item_fields().items():
            group = f.metadata.get("group", "")
            d[group].append((k, v, f))

        if order is None:
            return {group: sorted(kvf) for group, kvf in d.items()}

        return {group: sorted(d[group]) for group in order}


@contextmanager
def cmd_generator(
    name: str,
    *,
    subparsers: argparse._SubParsersAction,
) -> Iterator[ArgumentParser]:
    yield subparsers.add_parser(name, formatter_class=TypeHelpFormatter)
