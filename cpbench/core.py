"""The two building blocks shared by every configurable piece of cpbench (data components, feature learners,
detectors).

Component   parameters declared in `defaults` (merged along the class hierarchy), validated (unknown keys raise),
            stored in `params` and set as attributes; init() runs afterwards; spec() returns {type, **params}.
Registry    name -> (class, preset parameters).  @reg.register("name") registers a class, reg.preset(...) a named
            configuration of a registered class, reg.make("name", **params) or reg.make({"type": ..., ...}) builds
            an instance whose `name` is the registered name.
"""


def merged_defaults(cls):
    out = {}
    for c in reversed(cls.__mro__):
        out.update(c.__dict__.get("defaults", {}))
    return out


class Component:
    defaults = {}
    name = None

    def __init__(self, **params):
        known = merged_defaults(type(self))
        unknown = set(params) - set(known)
        if unknown:
            raise TypeError(f"{type(self).__name__}: unknown parameters {sorted(unknown)}; known: {sorted(known)}")
        self.params = {**known, **params}
        for k, v in self.params.items():
            setattr(self, k, v)
        self.init()

    def init(self):
        pass

    def spec(self):
        return dict(type=self.name, **self.params)

    def label(self):
        """Readable name: the registered name plus the parameters that differ from its (preset) defaults."""
        base = getattr(self, "_defaults", merged_defaults(type(self)))
        diff = {k: v for k, v in self.params.items() if base.get(k, object()) != v}
        fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v)
        return self.name + (f"({', '.join(f'{k}={fmt(v)}' for k, v in diff.items())})" if diff else "")

    def __repr__(self):
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


class Registry:
    def __init__(self, kind):
        self.kind, self.entries = kind, {}

    def register(self, name, **preset):
        def deco(cls):
            if name in self.entries and self.entries[name][0] is not cls:
                raise KeyError(f"{self.kind} {name!r} already registered")
            self.entries[name] = (cls, dict(preset))
            cls.name = name
            return cls
        return deco

    def preset(self, name, base, **params):
        cls, preset = self.entries[base]
        self.entries[name] = (cls, {**preset, **params})

    def make(self, spec, **params):
        """spec: a registered name, or a dict {"type": name, **params}."""
        if isinstance(spec, dict):
            spec = dict(spec); params = {**spec, **params}; spec = params.pop("type")
        if spec not in self.entries:
            raise KeyError(f"unknown {self.kind} {spec!r}; available: {', '.join(sorted(self.entries))}")
        cls, preset = self.entries[spec]
        obj = cls(**{**preset, **params})
        obj.name, obj._defaults = spec, {**merged_defaults(cls), **preset}
        return obj

    def defaults(self, name):
        cls, preset = self.entries[name]
        return {**merged_defaults(cls), **preset}

    def available(self, where=None):
        return sorted(n for n, (cls, _) in self.entries.items() if where is None or where(cls))

    def __contains__(self, name):
        return name in self.entries
