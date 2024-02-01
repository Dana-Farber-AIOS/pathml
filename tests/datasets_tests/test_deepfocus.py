from unittest.mock import MagicMock, mock_open, patch

import pytest

from pathml.datasets.deepfocus import DeepFocusDataModule


# Mocking the dataset for integrity check
@pytest.fixture
def mock_dataset(tmp_path):
    dataset_path = tmp_path / "outoffocus2017_patches5Classification.h5"
    dataset_path.write_bytes(b"fake content to simulate an actual file")
    return tmp_path


@patch("pathml.datasets.deepfocus.download_from_url")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=b"fake content")
@patch("hashlib.md5")
def test_deepfocusdatamodule_init_with_correct_checksum(
    mock_md5, mock_file, mock_exists, mock_download, mock_dataset
):
    # Setup mock to return a specific checksum
    mock_md5.return_value.hexdigest.return_value = "ba7b4a652c2a5a7079b216edd267b628"

    # Initialize the data module
    dm = DeepFocusDataModule(str(mock_dataset), download=False)

    # Ensure download was not triggered
    mock_download.assert_not_called()

    # Check if the integrity check passes
    assert dm._check_integrity()


# Test for initialization failure due to incorrect checksum
@patch("pathml.datasets.deepfocus.download_from_url")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=b"incorrect content")
@patch("hashlib.md5")
def test_deepfocusdatamodule_init_with_incorrect_checksum(
    mock_md5, mock_file, mock_exists, mock_download, mock_dataset
):
    # Setup mock to return an incorrect checksum
    mock_md5.return_value.hexdigest.return_value = "wrongchecksum"

    # Expect an AssertionError due to integrity check failure
    with pytest.raises(AssertionError):
        DeepFocusDataModule(str(mock_dataset), download=False)


# Test for automatic download when dataset is missing or fails integrity check
@patch("pathml.datasets.deepfocus.download_from_url")
@patch("os.path.exists", return_value=False)  # Simulate missing file
def test_deepfocusdatamodule_auto_download(mock_exists, mock_download, mock_dataset):
    DeepFocusDataModule(str(mock_dataset), download=True)

    # Verify that download_from_url was called due to missing dataset
    mock_download.assert_called_once_with(
        "https://zenodo.org/record/1134848/files/outoffocus2017_patches5Classification.h5",
        mock_dataset,
    )


@patch(
    "pathml.datasets.deepfocus.DeepFocusDataModule._check_integrity", return_value=True
)
def test_get_dataset(mock_check_integrity, mock_dataset):
    with patch("pathml.datasets.deepfocus.DeepFocusDataset") as mock_deepfocusdataset:
        dm = DeepFocusDataModule(str(mock_dataset), download=False)
        # Rest of your test code
        dm._get_dataset(fold_ix=1)
        mock_deepfocusdataset.assert_called_once_with(
            data_dir=mock_dataset, fold_ix=1, transforms=None
        )

        # Reset mock to test another fold index
        mock_deepfocusdataset.reset_mock()
        dm._get_dataset(fold_ix=2)
        mock_deepfocusdataset.assert_called_once_with(
            data_dir=mock_dataset, fold_ix=2, transforms=None
        )


@patch(
    "pathml.datasets.deepfocus.DeepFocusDataModule._check_integrity", return_value=True
)
@patch("pathml.datasets.deepfocus.logger.info")
def test_download_deepfocus_already_downloaded(
    mock_logger_info, mock_check_integrity, mock_dataset
):
    dm = DeepFocusDataModule(str(mock_dataset), download=True)
    dm._download_deepfocus(dm.data_dir)
    mock_logger_info.assert_called_with("File already downloaded with correct hash.")


# Ensure each mocked dataset returns the correct length for its respective fold
@patch(
    "pathml.datasets.deepfocus.DeepFocusDataModule._check_integrity", return_value=True
)
@patch("pathml.datasets.deepfocus.DeepFocusDataset")
def test_dataloader_properties(
    mock_deepfocusdataset, mock_check_integrity, mock_dataset
):
    # Mock lengths for train, validation, and test datasets respectively
    mock_deepfocusdataset.side_effect = [
        MagicMock(__len__=MagicMock(return_value=163199)),  # Training
        MagicMock(__len__=MagicMock(return_value=20400)),  # Validation
        MagicMock(__len__=MagicMock(return_value=20400)),  # Test
    ]

    dm = DeepFocusDataModule(
        str(mock_dataset), download=False, shuffle=True, batch_size=8
    )

    assert (
        len(dm.train_dataloader.dataset) == 163199
    ), "Incorrect length for training dataset"
    assert (
        len(dm.valid_dataloader.dataset) == 20400
    ), "Incorrect length for validation dataset"
    assert len(dm.test_dataloader.dataset) == 20400, "Incorrect length for test dataset"
