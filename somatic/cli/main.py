"""Computer Soup CLI — run an LLM too big for one machine, split across your machines."""

import typer

from somatic.cli.cluster_cmds import (
    cluster_bench,
    cluster_down,
    cluster_provision,
    cluster_ps,
    cluster_run,
    cluster_verify,
)

app = typer.Typer(
    help="Run an LLM too big for one machine, split across the machines you have.",
    no_args_is_help=True,
)

app.command("run")(cluster_run)
app.command("up", help="Alias of `run` — soup up your machines.")(cluster_run)
app.command("down")(cluster_down)
app.command("ps")(cluster_ps)
app.command("verify")(cluster_verify)
app.command("provision")(cluster_provision)
app.command("bench")(cluster_bench)


if __name__ == "__main__":
    app()
