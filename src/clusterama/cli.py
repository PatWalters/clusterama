import click

from .butina_cluster import run


@click.command()
@click.argument('path')
@click.argument('output')
@click.option('--cutoff', type=float, default=0.7)
def main(path, output, cutoff):
    run(path, output, cutoff)


if __name__ == "__main__":
    main()
