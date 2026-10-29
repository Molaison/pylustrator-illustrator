"""Read-only detection boundary; editor history remains the mutation owner."""

from typing import Annotated

from fastapi import Body

from .content_regions import detect_element_regions
from .document import DocumentError, new_document, validate_document


def install_content_region_routes(app, store):
    @app.post("/api/projects/{project_id}/content-regions/detect")
    def detect(project_id: str, body: Annotated[dict, Body()]):
        store.get_project(project_id)
        # Accept the current, possibly unsaved panel without saving the document.
        document = new_document()
        element = body.get("element")
        if not isinstance(element, dict):
            raise DocumentError("Choose an image or plot panel to detect.")
        document["elements"] = [{**element, "groupId": None}]
        element = validate_document(document)["elements"][0]
        return {"elementId": element["id"], "regions": detect_element_regions(
            element, lambda did: store.dataset_rows(project_id, did),
            lambda aid: store.get_asset(project_id, aid)[0])}
