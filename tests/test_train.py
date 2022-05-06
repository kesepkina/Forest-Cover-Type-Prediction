from pathlib import Path
from click.testing import CliRunner
import pytest
import joblib

from forest_cover_type.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_default_knn(
    runner: CliRunner,
    tmp_path
) -> None:
    """Testing default train knn call."""
    sample_path = Path('tests/train_sample.csv').resolve()
    with runner.isolated_filesystem(tmp_path):
        result = runner.invoke(
            train,
            [
                "knn",
                '-d',
                sample_path,
                '-s',
                tmp_path / 'test_model.joblib'
            ],
        )
        print(joblib.load(tmp_path / 'test_model.joblib'))
    assert result.exit_code == 0
    assert f"Model is saved to {tmp_path}/test_model.joblib." in result.output
    assert "nan" not in result.output

def test_error_forest_criterion(
    runner: CliRunner
) -> None:
    '''Testing error when wrong criterion inputed'''
    result = runner.invoke(
        train,
        [
            'forest',
            '--criterion',
            'jini'
        ]
    )
    assert result.exit_code == 2
    assert "Invalid value for '--criterion': 'jini' is not one of 'gini', 'entropy'." in result.output