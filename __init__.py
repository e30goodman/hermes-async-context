"""Async Context Engine — plugin registration.

Hermes loader contract: discovery finds the plugin by plugin.yaml in its directory,
imports the module and calls ``register(PluginContext)`` (plugins_loader).
Without register() the plugin is not loaded ("no register() function").
"""


def register_plugin(ctx):
    from .logsetup import setup_logging

    setup_logging()

    from .engine import AsyncContextEngine

    ctx.register_context_engine(AsyncContextEngine())


def register(ctx):
    """Plugin loader entry point (register, not register_plugin)."""
    register_plugin(ctx)