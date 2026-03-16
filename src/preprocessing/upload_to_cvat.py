"""
Upload excavator videos to CVAT (app.cvat.ai) as tasks under one project.
Usage (email/password):
    python src/upload_to_cvat.py --user YOUR_EMAIL --password YOUR_PASSWORD
Usage (API token, for OAuth accounts):
    python src/upload_to_cvat.py --token YOUR_API_TOKEN
"""

import argparse
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType
from cvat_sdk.models import ProjectWriteRequest, TaskWriteRequest

CVAT_HOST = "https://app.cvat.ai"
VIDEOS_DIR = Path("dataset/excavator-video/videos")
PROJECT_NAME = "excavator-video"
LABELS = [
    {"name": "excavator"},
    {"name": "person"},
    {"name": "truck"},
]


def main(user: str = None, password: str = None, token: str = None, project_id: int = None):
    if token:
        client_kwargs = {"access_token": token}
    else:
        client_kwargs = {"credentials": (user, password)}

    with make_client(host=CVAT_HOST, **client_kwargs) as client:
        # Use existing project or create new one
        if project_id:
            project = client.projects.retrieve(project_id)
            print(f"Using existing project '{project.name}' (id={project.id})")
        else:
            project = client.projects.create(
                ProjectWriteRequest(
                    name=PROJECT_NAME,
                    labels=LABELS,
                )
            )
            print(f"Created project '{PROJECT_NAME}' (id={project.id})")

        # Upload each video as a separate task
        video_files = sorted(VIDEOS_DIR.rglob("*.mp4"))
        print(f"Found {len(video_files)} videos. Uploading...")

        for video_path in video_files:
            task_name = video_path.stem
            print(f"  Uploading task: {task_name}")
            task = client.tasks.create_from_data(
                spec=TaskWriteRequest(
                    name=task_name,
                    project_id=project.id,
                ),
                resource_type=ResourceType.LOCAL,
                resources=[str(video_path)],
            )
            print(f"    -> Task id={task.id}")

        print("Done! Open https://app.cvat.ai to start labeling.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", help="CVAT account email (not needed with --token)")
    parser.add_argument("--password", help="CVAT account password (not needed with --token)")
    parser.add_argument("--token", help="CVAT API token (for OAuth accounts)")
    parser.add_argument("--project-id", type=int, help="Use an existing project by ID instead of creating a new one")
    args = parser.parse_args()

    if not args.token and not (args.user and args.password):
        parser.error("Provide either --token or both --user and --password")

    main(user=args.user, password=args.password, token=args.token, project_id=args.project_id)
