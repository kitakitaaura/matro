export interface EvalResult {
  ok: boolean;
  output: string;
}

export interface WorkspaceVar {
  name: string;
  size: string;
  class: string;
  value: string;
}

export interface PlotSeriesJson {
  kind: "line" | "scatter" | "bar" | "stem";
  color: string;
  dashed: boolean;
  label: string | null;
  x: (number | null)[];
  y: (number | null)[];
}

export interface PlotPanelJson {
  title: string;
  xlabel: string;
  ylabel: string;
  grid: boolean;
  has_heatmap: boolean;
  series: PlotSeriesJson[];
}

export interface PlotFigureJson {
  rows: number;
  cols: number;
  panels: PlotPanelJson[];
}

export class MatroSession {
  eval(code: string): EvalResult;
  disasm(code: string): EvalResult;
  mountFile(path: string, content: string): EvalResult;
  mountBytes(path: string, bytes: Uint8Array | ArrayBuffer): EvalResult;
  files(): string[];
  clearFiles(): void;
  setStepLimit(limit: number | bigint): void;
  vars(): WorkspaceVar[];
  plotJson(): PlotFigureJson | null;
  plotSvg(width?: number, height?: number): string;
  reset(): void;
}

export type WasmSource = string | URL | ArrayBuffer | ArrayBufferView | Response;

export function loadMatro(source?: WasmSource): Promise<MatroSession>;
