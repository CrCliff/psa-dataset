from typing import Dict, Tuple
import boto3
import time

START=0
STOP=60

SUBNETS = ["subnet-1f26da53", "subnet-47113d21"]
SECURITY_GROUPS = ["sg-0ce615b54d6fb23c1"]
ECS_CLUSTER = "arn:aws:ecs:us-east-1:027517924056:cluster/psa-process"
ECS_TASK_DEFINITION = "psa-process"
S3_BUCKET = "psa-dataset"
S3_PREFIX_IN = "raw"
S3_PREFIX_OUT = "processed"


def s3_urls(i: int) -> Tuple[str, str]:
    sub = (i // 100) * 100
    return (
        f"s3://{S3_BUCKET}/{S3_PREFIX_IN}/{sub:04}/{i:04}.csv",
        f"s3://{S3_BUCKET}/{S3_PREFIX_OUT}/{sub:04}/{i:04}.csv",
    )


def get_params(s3_in: str, s3_out: str) -> dict:
    return {
        "cluster": ECS_CLUSTER,
        "count": 1,
        "enableECSManagedTags": True,
        "enableExecuteCommand": False,
        "launchType": "FARGATE",
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": SUBNETS,
                "securityGroups": SECURITY_GROUPS,
                "assignPublicIp": "ENABLED",
            }
        },
        "overrides": {
            "containerOverrides": [
                {
                    "name": "psa-process",
                    "environment": [
                        {"name": "S3_IN", "value": s3_in},
                        {
                            "name": "S3_OUT",
                            "value": s3_out,
                        },
                    ],
                }
            ]
        },
        "tags": [
            {
                "key": "S3_IN",
                "value": s3_in,
            },
            {
                "key": "S3_OUT",
                "value": s3_out,
            },
        ],
        "propagateTags": "TASK_DEFINITION",
        "taskDefinition": ECS_TASK_DEFINITION,
    }


if __name__ == "__main__":
    ecs = boto3.client("ecs", region_name="us-east-1")

    for i in range(START, STOP):
        s3_in, s3_out = s3_urls(i)

        params = get_params(s3_in, s3_out)

        resp = ecs.run_task(**params)
        
        print(i, resp)

        if i != 0 and i % 49 == 0:
            # We can only run 50 tasks concurrently, wait for these to finish
            print(f'Waiting on task {i}...')
            time.sleep(240)
