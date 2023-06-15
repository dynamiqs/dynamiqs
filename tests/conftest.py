import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        '--gpu', action='store_true', help='Set default PyTorch device to "cuda"'
    )


@pytest.fixture(scope='session', autouse=True)
def device(request):
    use_gpu = request.config.getoption('--gpu')
    if use_gpu:
        torch.set_default_device('cuda')
