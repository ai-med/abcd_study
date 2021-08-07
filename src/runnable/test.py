import sys
import click

@click.command()
@click.option('--shout', is_flag=True)
def main(shout):
    rv = sys.platform
    if shout:
        rv = rv.upper() + '!!!!111'
    click.echo(shout)

if __name__ == '__main__':
    main()