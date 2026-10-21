# Web workspace implementation contract

This is the shared implementation contract for the Web migration. The baseline is
PR #14 (`369f5d1`), which already includes `main`. Desktop remains an optional adapter.

## Ownership

- `web/document.py`, `web/render.py`, `web/session.py`, package lazy imports: headless agent.
- `web/store.py`, `web/datasets.py`, `web/ai.py`: research-data agent.
- `web/static/`: browser-workspace agent.
- `web/app.py`, `web/__main__.py`, `web/protein.py`, packaging, integration: root.

## Browser API

All JSON errors use `{ "error": { "code": string, "message": string } }`.
No endpoint accepts executable Python. API is same-origin and defaults to loopback.

| Method | Path | Request / response |
|---|---|---|
| GET | `/api/health` | `{status, version}` |
| GET | `/api/projects` | `{projects: [{id,name,revision,updatedAt}]}` |
| POST | `/api/projects` | `{name,description?,example?:boolean}` → full project |
| GET | `/api/projects/{id}` | full project |
| PUT | `/api/projects/{id}` | `{revision,document?,name?,description?,tags?}` → full project; 409 on stale revision |
| POST | `/api/projects/import` | exported project bundle JSON → new full project |
| GET | `/api/projects/{id}/bundle` | downloadable project bundle JSON |
| GET | `/api/projects/{id}/versions` | `{versions:[{id,name,createdAt,revision,...}]}` |
| POST | `/api/projects/{id}/versions` | `{name,revision}` → version |
| POST | `/api/projects/{id}/versions/{versionId}/restore` | `{revision}` → full project; creates recoverable pre-restore checkpoint |
| POST | `/api/projects/{id}/datasets` | `{name,format:'csv'|'tsv'|'json',content:string}` → full project |
| GET | `/api/projects/{id}/datasets/{datasetId}?offset=0&limit=100` | `{id,name,columns:[{name,type,missing}],rows:[object],rowCount,offset,limit,hash,sourceId?,recipe?}` |
| POST | `/api/projects/{id}/datasets/{datasetId}/preview` | `{steps:[...]}` → preview as above; no mutation |
| POST | `/api/projects/{id}/datasets/{datasetId}/transform` | `{steps:[...],name?,expectedHash?}` → full project; new derived immutable dataset |
| GET | `/api/providers` | `{providers:[{id,label,configured,model}],operations:[...]}`; no secrets |
| POST | `/api/projects/{id}/datasets/{datasetId}/ai-plan` | `{provider,prompt}` → `{id,provider,model,inputHash,steps,explanation}`; metadata only; data rows are never sent |
| POST | `/api/projects/{id}/structures` | `{name,format:'pdb'|'cif',content}` → full project |
| GET | `/api/projects/{id}/structures/{structureId}` | structure metadata, source `content`, format, chains, atomCount, view |
| PUT | `/api/projects/{id}/structures/{structureId}` | `{view:{representation,color,chain,camera},revision}` → full project |
| POST | `/api/projects/{id}/assets` | `{name,mimeType:'image/png',data:base64,provenance?}` → `{asset,project}` |
| GET | `/api/projects/{id}/assets/{assetId}` | image bytes |
| POST | `/api/render/plot` | `{projectId,plot,width?,height?}` → SVG image |
| POST | `/api/projects/{id}/export` | `{format:'svg'|'png'|'pdf'|'python',document?}` → download |

Full project: `{id,name,description,tags,revision,createdAt,updatedAt,document,datasets,structures,assets}`.
Dataset/structure/asset list entries contain metadata, never unbounded raw content.
Dataset/structure imports and transforms return a full project and advance its revision.
The UI must first save dirty document state before adding an asset to avoid losing edits.

## Document schema (version 1)

```json
{
  "schemaVersion": 1,
  "id": "figure-1", "name": "Figure 1", "width": 960, "height": 640,
  "background": "#ffffff", "groups": [],
  "elements": [
    {
      "id": "element-1", "type": "text", "name": "Panel label",
      "x": 40, "y": 30, "width": 180, "height": 40,
      "rotation": 0, "visible": true, "locked": false, "groupId": null,
      "style": {"fill":"#1f2937","stroke":"none","strokeWidth":1,"opacity":1,"fontSize":24},
      "text": "A"
    }
  ]
}
```

Coordinate origin is top left; units are pixels at 96 DPI. Elements are in paint order.
Types: text, rect, ellipse, line, path, plot, image. Path/line `points` are local [x,y]
pairs. A plot has `plot:{datasetId,kind,x,y,title,xlabel,ylabel,color}`;
kind is line, scatter, bar, or histogram. An image has `assetId`. Group entries are
`{id,name}`. Group membership is flat and explicit. Unsupported types are rejected.
Shapes never execute scripts. Renderers must resolve assets within their project.

## Interaction and integrity

V selection, A point selection, Shift multi-select, Alt overlap cycling, Space pan,
Ctrl/Cmd+Z undo, Ctrl/Cmd+Shift+Z redo, Ctrl/Cmd+S save, arrow nudge, Delete,
Ctrl/Cmd+G group and Shift variant ungroup. Editing fields suppress canvas shortcuts.
One gesture commits one history item; Escape/pointercancel rolls it back.
Original datasets are immutable; transformations produce derived datasets with hashes.
AI returns a restricted transform recipe; preview and apply are separate user actions.
Working undo and durable project versions are separate concepts.

## Native Matplotlib compatibility

A separate FigureSession accepts a trusted in-process Figure, exposes stable artist IDs,
bounds and supported operations, executes adapter-owned commands, and renders using Agg.
This preserves existing scientific semantic editing without accepting uploaded Python
or pickles. The static document workflow and native figure sessions are both headless.

Native session endpoints: `GET /api/sessions` (lightweight `{id,name,revision}` list,
without artist geometry), `GET /api/sessions/{id}` (full editable artist description),
`POST /api/sessions/{id}/commands`, `GET /api/sessions/{id}/render.svg` (PNG/PDF
also accepted), and `GET /api/sessions/{id}/changes.py`. The changes snippet
requires the original `fig` construction and rejects a different initial size/DPI.
