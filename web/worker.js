import { MatroRuntime } from "./matro.js";

let rt = null;
const ready = MatroRuntime.load("./matro_wasm.wasm").then((runtime) => {
  rt = runtime;
  self.postMessage({ type: "ready" });
}).catch((err) => {
  self.postMessage({ type: "init_error", error: String(err) });
});

self.onmessage = async (e) => {
  await ready;
  if (!rt) return;
  const msg = e.data;
  const id = msg.id;

  try {
    if (msg.type === "run_script") {
      rt.reset();
      const t0 = performance.now();
      const res = rt.evalFile(msg.path || "", msg.code);
      const dt = performance.now() - t0;
      const dis = rt.disasm(msg.code);
      self.postMessage({
        id,
        type: "result",
        ok: res.ok,
        output: res.output,
        dt,
        disasm: dis.output,
        vars: rt.vars(),
        files: rt.files(),
        svg: rt.plotSvg(600, 280),
      });
    } else if (msg.type === "repl") {
      const res = rt.eval(msg.code);
      self.postMessage({
        id,
        type: "repl_result",
        ok: res.ok,
        output: res.output,
        vars: rt.vars(),
        files: rt.files(),
        svg: rt.plotSvg(600, 280),
      });
    } else if (msg.type === "mount_file") {
      rt.mountFile(msg.path, msg.content);
      self.postMessage({ id, type: "state", vars: rt.vars(), files: rt.files(), svg: rt.plotSvg(600, 280) });
    } else if (msg.type === "mount_bytes") {
      rt.mountBytes(msg.path, new Uint8Array(msg.bytes));
      self.postMessage({ id, type: "state", vars: rt.vars(), files: rt.files(), svg: rt.plotSvg(600, 280) });
    } else if (msg.type === "clear_files") {
      rt.clearFiles();
      self.postMessage({ id, type: "state", vars: rt.vars(), files: rt.files(), svg: rt.plotSvg(600, 280) });
    } else if (msg.type === "reset") {
      rt.reset();
      self.postMessage({ id, type: "state", vars: rt.vars(), files: rt.files(), svg: "" });
    }
  } catch (err) {
    self.postMessage({ id, type: "result", ok: false, output: String(err), dt: 0, disasm: "", vars: [], files: [], svg: "" });
  }
};
