from typing import Any
import boto3


class S3Reader:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.is_open = False
        self.swap_file = None
        self.file_name = None
        self.f = None

    def _read(self, file: str, bucket: str, key: str) -> None:
        self.s3.download_file(
            Bucket=bucket,
            Key=key,
            Filename=file,
        )

    def read(self, file: str, bucket: str, key: str) -> None:
        if not isinstance(file, str):
            raise ValueError("file must be of type str")

        if not isinstance(bucket, str):
            raise ValueError("bucket must be of type str")

        if not isinstance(key, str):
            raise ValueError("key must be of type str")

        self._read(file, bucket, key)
