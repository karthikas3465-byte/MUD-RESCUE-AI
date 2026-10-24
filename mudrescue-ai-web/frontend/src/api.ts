export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep status */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const media = (missionId: string, kind: "source" | "frame" | "clip", revision = 0) =>
  `/api/missions/${missionId}/media/${kind}?v=${revision}`;

export const evidence = (detectionId: string) => `/api/detections/${detectionId}/evidence`;
