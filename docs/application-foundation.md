# Application foundation decision

The first application boundary uses a Python 3.11 package in `src/tkd_poomsae/` for local analysis and an empty FastAPI service. `tkd-poomsae serve` starts that service on loopback by default. The TypeScript/React client in `web/` is a separate static development build. Heavy video and model processing belongs outside the browser; this foundation makes no claim to implement it.

`uv.lock` and `web/package-lock.json` pin resolved core and client dependencies. The Python core has no PyTorch or MMPose dependency. A later, separately specified vision environment will resolve that stack. CI uses the frozen Python lock and `npm ci`; source, docs, and SkillForge automation have separate ownership.
