import boto3


class S3Writer:
    def __init__(self):
        self.s3 = boto3.client("s3")

    def _write(self, file: str, bucket: str, key: str) -> None:
        self.s3.put_object(
            ACL='bucket-owner-full-control',
            Body=open(file, 'rb'),
            Bucket=bucket,
            Key=key,
        )

    def write(self, file: str, bucket: str, key: str) -> None:
        if not isinstance(file, str):
            raise ValueError("file must be of type str")

        if not isinstance(bucket, str):
            raise ValueError("bucket must be of type str")

        if not isinstance(key, str):
            raise ValueError("key must be of type str")

        self._write(file, bucket, key)
