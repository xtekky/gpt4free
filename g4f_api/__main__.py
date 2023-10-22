import typing
from enum import Enum
from g4f import Provider
from g4f_api import ApiInterface
import typer

IgnoredProviders = Enum("ignore_providers", {key:key for key in Provider.__all__})

app = typer.Typer(help="Run the G4F API")

@app.command()
def main(
    bind_str: str = typer.Argument(..., envvar="G4F_API_BIND_STR", help="The bind string."),
    i_num_threads: int = typer.Option(1, envvar="G4F_API_NUM_THREADS", help="The number of threads."),
    list_ignored_providers: typing.List[IgnoredProviders] = typer.Option([], envvar="G4F_API_LIST_IGNORED_PROVIDERS", help="List of providers to ignore when processing request."),
):
	list_ignored_providers=[provider.name for provider in list_ignored_providers]
	ApiInterface.list_ignored_providers=list_ignored_providers
	ApiInterface.api.run(bind_str, i_num_threads)


if __name__ == "__main__":
	app()