import os
import supervisely as sly
from dotenv import load_dotenv


project_meta = sly.ProjectMeta()

#

exit(0)

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api.from_env()

workspace_id = int(os.environ["context.workspaceId"])
workspace = api.workspace.get_info_by_id(workspace_id)
if workspace is None:
    raise ValueError(
        f"Workspace (id={workspace_id}) not found. Put correct value to local.env"
    )
