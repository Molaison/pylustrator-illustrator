export class ApiError extends Error {
  constructor(message, status, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export async function request(
  path,
  { method = "GET", body, signal, binary = false } = {},
) {
  const timeout = AbortSignal.timeout(120000);
  let response;
  try {
    response = await fetch(path, {
      method,
      credentials: "same-origin",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(
      error.name === "TimeoutError"
        ? "The request timed out. Your current work is preserved; try again."
        : "The server could not be reached. Your current work is preserved; check the connection and try again.",
      0,
      "network",
    );
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.error?.message || payload.detail;
    throw new ApiError(
      typeof detail === "string"
        ? detail
        : `Request failed (${response.status}). Please try again.`,
      response.status,
      payload.error?.code,
    );
  }
  if (binary) return response.blob();
  if (response.status === 204) return null;
  return response.json();
}

export const api = {
  get: (path, options) => request(path, options),
  post: (path, body, options) =>
    request(path, { ...options, method: "POST", body }),
  put: (path, body, options) =>
    request(path, { ...options, method: "PUT", body }),
};

export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export const projectPath = (project, tail = "") =>
  `/api/projects/${encodeURIComponent(project.id)}${tail}`;
export const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const uid = (prefix) =>
  `${prefix}-${globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)}`;
export const prettyDate = (value) =>
  new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));

export async function busy(button, fn, label = "Working…") {
  if (button?.disabled) return;
  const original = button?.textContent;
  if (button) {
    button.disabled = true;
    button.textContent = label;
    button.setAttribute("aria-busy", "true");
  }
  try {
    return await fn();
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = original;
      button.removeAttribute("aria-busy");
    }
  }
}
