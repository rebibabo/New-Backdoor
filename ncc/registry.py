# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

REGISTRIES = {}


def setup_registry(
    registry_name: str,
    base_class=None,
    default=None,
):
    registry_name = registry_name.replace('-', '_')

    REGISTRY = {}
    REGISTRY_CLASS_NAMES = set()

    # maintain a registry of all registries
    if registry_name in REGISTRIES:
        return  # registry already exists
    REGISTRIES[registry_name] = {
        'registry': REGISTRY,
        'default': default,
    }

    def build_x(args, *extra_args, **extra_kwargs):
        choice = args[registry_name] if registry_name in args else None
        if choice is None:
            return None
        cls = REGISTRY[choice]
        if hasattr(cls, 'build_' + registry_name):
            builder = getattr(cls, 'build_' + registry_name)
        else:
            builder = cls
        set_defaults(args, cls)
        return builder(args, *extra_args, **extra_kwargs)

    def register_x(name):

        def register_x_cls(cls):
            if name in REGISTRY:
                raise ValueError('Cannot register duplicate {} ({})'.format(registry_name, name))
            if cls.__name__ in REGISTRY_CLASS_NAMES:
                raise ValueError(
                    'Cannot register {} with duplicate class name ({})'.format(
                        registry_name, cls.__name__,
                    )
                )
            if base_class is not None and not issubclass(cls, base_class):
                raise ValueError('{} must extend {}'.format(cls.__name__, base_class.__name__))
            REGISTRY[name] = cls
            REGISTRY_CLASS_NAMES.add(cls.__name__)
            return cls

        return register_x_cls

    return build_x, register_x, REGISTRY


def set_defaults(*args, **kwargs):
    """Helper to set default arguments based on *add_args*."""
    pass
