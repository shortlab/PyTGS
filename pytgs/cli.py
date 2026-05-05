import yaml

from pytgs.analyzer import TGSAnalyzer


def main() -> None:
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    analyzer = TGSAnalyzer(config)
    analyzer.fit(show=True)


if __name__ == '__main__':
    main()
