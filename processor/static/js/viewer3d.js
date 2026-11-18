/*
===========================================================
 Gurukrupa Exports Pvt Ltd
 3D Jewellery Viewer

 Turns a cutout + height map (from /make-3d/) into a relief
 mesh you can rotate, light, and export (PNG, GLB, video,
 360-degree frame set).
===========================================================
*/
import {
    ACESFilmicToneMapping,
    BufferAttribute,
    CanvasTexture,
    Color,
    DirectionalLight,
    FrontSide,
    Group,
    LinearFilter,
    LinearMipmapLinearFilter,
    Mesh,
    MeshBasicMaterial,
    MeshPhysicalMaterial,
    NeutralToneMapping,
    PerspectiveCamera,
    PlaneGeometry,
    PMREMGenerator,
    Scene,
    ShadowMaterial,
    SRGBColorSpace,
    Vector2,
    WebGLRenderer,
    OrbitControls,
    GLTFExporter,
    RoomEnvironment,
} from "../vendor/three/three-bundle.min.js";

const MATERIALS = {
    studio:  { label: "Studio",          metalness: 0.15, roughness: 0.45, env: 1.0, clearcoat: 0.2 },
    metal:   { label: "Polished metal",  metalness: 0.6,  roughness: 0.22, env: 1.5, clearcoat: 0.6 },
    matte:   { label: "Soft matte",      metalness: 0.0,  roughness: 0.8,  env: 0.8, clearcoat: 0.0 },
    photo:   { label: "Photo-true (unlit)" },
};

const $ = (id) => document.getElementById(id);

export class JewelleryViewer {
    constructor(container, { onStatus } = {}) {
        this.container = container;
        this.onStatus = onStatus || (() => {});
        this.data = null;
        this.settings = {
            depth: 35,           // % of object width
            detail: 30,          // fine relief strength (normal map)
            smooth: 1,           // blur passes on the height grid
            resolution: 256,     // mesh segments along the long side
            material: "studio",
            light: 1.0,
            exposure: 1.0,
            back: "mirror",      // mirror | none
            motion: "swing",     // swing | spin | off
            loopSeconds: 6,
            swingDegrees: 35,
            shadow: true,
            background: "#f4f6fb",
            transparent: false,
            wireframe: false,
        };
        this.motionTime = 0;
        this.clockLast = performance.now();
        this._initThree();
    }

    // ------------------------------------------------------------
    // Setup
    // ------------------------------------------------------------
    _initThree() {
        const renderer = new WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        renderer.outputColorSpace = SRGBColorSpace;
        renderer.toneMapping = NeutralToneMapping ?? ACESFilmicToneMapping;
        renderer.shadowMap.enabled = true;
        this.container.appendChild(renderer.domElement);
        this.renderer = renderer;

        const scene = new Scene();
        this.scene = scene;
        const pmrem = new PMREMGenerator(renderer);
        scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

        const camera = new PerspectiveCamera(30, 1, 0.01, 100);
        camera.position.set(0, 0.15, 5.4);
        this.camera = camera;

        const key = new DirectionalLight(0xffffff, 1.6);
        key.position.set(2.5, 3.5, 4);
        key.castShadow = true;
        key.shadow.mapSize.set(2048, 2048);
        key.shadow.camera.left = key.shadow.camera.bottom = -2;
        key.shadow.camera.right = key.shadow.camera.top = 2;
        key.shadow.bias = -0.0005;
        key.shadow.radius = 6;
        scene.add(key);
        this.keyLight = key;

        const ground = new Mesh(new PlaneGeometry(12, 12), new ShadowMaterial({ opacity: 0.18 }));
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -1.08;
        ground.receiveShadow = true;
        scene.add(ground);
        this.ground = ground;

        this.pivot = new Group();       // turntable motion
        this.model = new Group();       // front + back meshes
        this.pivot.add(this.model);
        scene.add(this.pivot);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 1.2;
        controls.maxDistance = 12;
        controls.target.set(0, 0, 0);
        this.controls = controls;

        this._resizeObserver = new ResizeObserver(() => this.resize());
        this._resizeObserver.observe(this.container);
        this.resize();
        this.applyBackground();

        const loop = () => {
            this._raf = requestAnimationFrame(loop);
            const now = performance.now();
            const dt = Math.min((now - this.clockLast) / 1000, 0.1);
            this.clockLast = now;
            if (!this._recording) {
                this.motionTime += dt;
                this._applyMotion(this.motionTime);
            }
            controls.update();
            renderer.render(scene, camera);
        };
        loop();
    }

    resize() {
        const w = this.container.clientWidth || 640;
        const h = this.container.clientHeight || 480;
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
    }

    _applyMotion(t) {
        const s = this.settings;
        const loop = Math.max(1, s.loopSeconds);
        if (s.motion === "spin") {
            this.pivot.rotation.y = ((t % loop) / loop) * Math.PI * 2;
        } else if (s.motion === "swing") {
            this.pivot.rotation.y = Math.sin((t / loop) * Math.PI * 2) * (s.swingDegrees * Math.PI / 180);
        }
    }

    // ------------------------------------------------------------
    // Data loading
    // ------------------------------------------------------------
    async load(payload) {
        const [cutout, depthImg, detailImg] = await Promise.all([
            loadImage("data:image/png;base64," + payload.cutout),
            loadImage("data:image/png;base64," + payload.depth),
            loadImage("data:image/png;base64," + payload.detail),
        ]);
        const w = payload.width, h = payload.height;
        const depthPx = readPixels(depthImg, w, h);
        const detailPx = readPixels(detailImg, w, h);
        const depth = new Float32Array(w * h);
        const detail = new Float32Array(w * h);
        for (let i = 0, p = 0; i < w * h; i++, p += 4) {
            depth[i] = ((depthPx[p] << 8) | depthPx[p + 1]) / 65535;
            detail[i] = (detailPx[p] - 128) / 127;
        }
        this.data = { name: payload.name || "jewellery", cutout, depth, detail, w, h };

        if (this.colorTexture) this.colorTexture.dispose();
        const tex = new CanvasTexture(imageToCanvas(cutout));
        tex.colorSpace = SRGBColorSpace;
        tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
        tex.minFilter = LinearMipmapLinearFilter;
        tex.magFilter = LinearFilter;
        this.colorTexture = tex;

        this.rebuild();
        this.resetView();
    }

    // ------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------
    rebuild() {
        if (!this.data) return;
        const { w, h, depth } = this.data;
        const s = this.settings;
        const aspect = w / h;
        const width = aspect >= 1 ? 2 : 2 * aspect;
        const height = aspect >= 1 ? 2 / aspect : 2;
        const segX = Math.max(8, Math.round(aspect >= 1 ? s.resolution : s.resolution * aspect));
        const segY = Math.max(8, Math.round(aspect >= 1 ? s.resolution / aspect : s.resolution));

        // sample + smooth the height grid
        let grid = new Float32Array((segX + 1) * (segY + 1));
        for (let j = 0; j <= segY; j++) {
            const v = j / segY;
            for (let i = 0; i <= segX; i++) {
                grid[j * (segX + 1) + i] = sampleBilinear(depth, w, h, i / segX, v);
            }
        }
        for (let p = 0; p < s.smooth; p++) grid = blurGrid(grid, segX + 1, segY + 1);

        const zMax = (s.depth / 100) * Math.max(width, height) * 0.5;
        const front = new PlaneGeometry(width, height, segX, segY);
        const pos = front.attributes.position;
        // PlaneGeometry rows go top -> bottom, same as image rows.
        for (let k = 0; k < pos.count; k++) pos.setZ(k, grid[k] * zMax);
        front.computeVertexNormals();

        const back = front.clone();
        const bpos = back.attributes.position;
        for (let k = 0; k < bpos.count; k++) bpos.setZ(k, -grid[k] * zMax);
        flipWinding(back);
        back.computeVertexNormals();

        this._disposeModel();
        const frontMat = this._makeMaterial();
        const frontMesh = new Mesh(front, frontMat);
        frontMesh.name = "front";
        frontMesh.castShadow = true;
        this.model.add(frontMesh);

        if (s.back === "mirror") {
            const backMesh = new Mesh(back, this._makeMaterial());
            backMesh.name = "back";
            backMesh.castShadow = true;
            this.model.add(backMesh);
        } else {
            back.dispose();
        }
        this.ground.position.y = -height / 2 - 0.08;
        this.ground.visible = s.shadow && !s.transparent;
        this.keyLight.castShadow = s.shadow;
        this.objectSize = { width, height, depth: zMax * 2 };
        this.updateNormalMap();
    }

    _makeMaterial() {
        const s = this.settings;
        const preset = MATERIALS[s.material] || MATERIALS.studio;
        if (s.material === "photo") {
            return new MeshBasicMaterial({
                map: this.colorTexture, alphaTest: 0.5, side: FrontSide, wireframe: s.wireframe,
            });
        }
        const mat = new MeshPhysicalMaterial({
            map: this.colorTexture,
            alphaTest: 0.5,
            side: FrontSide,
            metalness: preset.metalness,
            roughness: preset.roughness,
            clearcoat: preset.clearcoat,
            clearcoatRoughness: 0.15,
            envMapIntensity: preset.env * s.light,
            wireframe: s.wireframe,
        });
        if (this.normalTexture) {
            mat.normalMap = this.normalTexture;
            mat.normalScale = new Vector2(1, 1);
        }
        return mat;
    }

    updateNormalMap() {
        if (!this.data) return;
        const { detail, w, h } = this.data;
        const strength = this.settings.detail / 100;
        const scale = Math.min(1, 1024 / Math.max(w, h));
        const nw = Math.max(2, Math.round(w * scale));
        const nh = Math.max(2, Math.round(h * scale));
        const out = new Uint8Array(nw * nh * 4);
        const k = strength * 6;
        const at = (x, y) => detail[Math.min(h - 1, Math.max(0, Math.round(y / scale))) * w
            + Math.min(w - 1, Math.max(0, Math.round(x / scale)))];
        for (let y = 0; y < nh; y++) {
            for (let x = 0; x < nw; x++) {
                const dx = (at(x + 1, y) - at(x - 1, y)) * k;
                const dy = (at(x, y + 1) - at(x, y - 1)) * k;
                const len = Math.hypot(dx, dy, 1);
                // row 0 = top of the image, so image +y points down the surface
                const p = (y * nw + x) * 4;
                out[p] = ((-dx / len) * 0.5 + 0.5) * 255;
                out[p + 1] = ((dy / len) * 0.5 + 0.5) * 255;
                out[p + 2] = ((1 / len) * 0.5 + 0.5) * 255;
                out[p + 3] = 255;
            }
        }
        if (this.normalTexture) this.normalTexture.dispose();
        // Canvas texture (not DataTexture) so the GLB exporter can embed it.
        // Default flipY puts canvas row 0 (image top) at v=1, matching the plane UVs.
        const canvas = document.createElement("canvas");
        canvas.width = nw;
        canvas.height = nh;
        canvas.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(out.buffer), nw, nh), 0, 0);
        const tex = new CanvasTexture(canvas);
        tex.minFilter = LinearMipmapLinearFilter;
        tex.magFilter = LinearFilter;
        this.normalTexture = strength > 0 ? tex : null;
        if (strength <= 0) tex.dispose();
        this.model.traverse((obj) => {
            if (obj.isMesh && obj.material && "normalMap" in obj.material && this.settings.material !== "photo") {
                obj.material.normalMap = this.normalTexture;
                obj.material.needsUpdate = true;
            }
        });
    }

    _disposeModel() {
        [...this.model.children].forEach((child) => {
            this.model.remove(child);
            child.geometry?.dispose();
            child.material?.dispose();
        });
    }

    // ------------------------------------------------------------
    // Look
    // ------------------------------------------------------------
    set(key, value) {
        this.settings[key] = value;
        const rebuildKeys = ["depth", "smooth", "resolution", "back", "material", "wireframe", "shadow"];
        if (rebuildKeys.includes(key)) this.rebuild();
        if (key === "detail") this.updateNormalMap();
        if (key === "light") {
            this.keyLight.intensity = 1.6 * value;
            this.model.traverse((o) => {
                if (o.isMesh && o.material && "envMapIntensity" in o.material) {
                    const preset = MATERIALS[this.settings.material] || MATERIALS.studio;
                    o.material.envMapIntensity = (preset.env || 1) * value;
                }
            });
        }
        if (key === "exposure") this.renderer.toneMappingExposure = value;
        if (key === "background" || key === "transparent") {
            this.applyBackground();
            this.ground.visible = this.settings.shadow && !this.settings.transparent;
        }
        if (key === "motion" && value === "off") {
            // keep the current angle; the user can orbit freely
        }
    }

    applyBackground() {
        if (this.settings.transparent) {
            this.scene.background = null;
            this.renderer.setClearColor(0x000000, 0);
            this.container.classList.add("is-transparent");
        } else {
            this.scene.background = new Color(this.settings.background);
            this.container.classList.remove("is-transparent");
        }
    }

    resetView() {
        this.camera.position.set(0, 0.15, 5.4);
        this.controls.target.set(0, 0, 0);
        this.pivot.rotation.set(0, 0, 0);
        this.motionTime = 0;
        this.controls.update();
    }

    // ------------------------------------------------------------
    // Export
    // ------------------------------------------------------------
    _renderAt(width, height, angle = null) {
        const r = this.renderer;
        const prevSize = r.getSize(new Vector2());
        const prevRatio = r.getPixelRatio();
        const prevAspect = this.camera.aspect;
        const prevRot = this.pivot.rotation.y;
        r.setPixelRatio(1);
        r.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        if (angle !== null) this.pivot.rotation.y = angle;
        r.render(this.scene, this.camera);
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        canvas.getContext("2d").drawImage(r.domElement, 0, 0, width, height);
        r.setPixelRatio(prevRatio);
        r.setSize(prevSize.x, prevSize.y, false);
        this.camera.aspect = prevAspect;
        this.camera.updateProjectionMatrix();
        this.pivot.rotation.y = prevRot;
        return canvas;
    }

    _exportSize(longSide) {
        const aspect = this.camera.aspect || 1;
        return aspect >= 1
            ? [longSide, Math.round(longSide / aspect)]
            : [Math.round(longSide * aspect), longSide];
    }

    async snapshotPNG(longSide = 2000) {
        const [w, h] = this._exportSize(longSide);
        const canvas = this._renderAt(w, h);
        const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
        download(blob, `${this.data?.name || "jewellery"}_3d.png`);
    }

    async exportGLB() {
        if (!this.data) return;
        const exporter = new GLTFExporter();
        // Export the static model (no turntable rotation).
        const clone = this.model.clone(true);
        const result = await exporter.parseAsync(clone, { binary: true, onlyVisible: true });
        download(new Blob([result], { type: "model/gltf-binary" }), `${this.data.name}_3d.glb`);
    }

    async recordVideo(onProgress) {
        if (!this.data) return;
        if (!window.MediaRecorder) throw new Error("This browser cannot record video.");
        const candidates = [
            ["video/mp4;codecs=avc1", "mp4"],
            ["video/mp4", "mp4"],
            ["video/webm;codecs=vp9", "webm"],
            ["video/webm", "webm"],
        ];
        const [mime, ext] = candidates.find(([m]) => MediaRecorder.isTypeSupported(m)) || ["", "webm"];
        // Record at 2x the on-screen size for a sharper video.
        const prevRatio = this.renderer.getPixelRatio();
        this.renderer.setPixelRatio(2);
        this.resize();
        const stream = this.renderer.domElement.captureStream(60);
        const recorder = new MediaRecorder(stream, { mimeType: mime || undefined, videoBitsPerSecond: 12_000_000 });
        const chunks = [];
        recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
        const done = new Promise((res) => (recorder.onstop = res));

        const motion = this.settings.motion === "off" ? "spin" : this.settings.motion;
        const prevMotion = this.settings.motion;
        this.settings.motion = motion;
        const duration = Math.max(1, this.settings.loopSeconds) * 1000;
        this._recording = true;
        const start = performance.now();
        recorder.start(250);
        await new Promise((resolve) => {
            const tick = () => {
                const t = performance.now() - start;
                this._applyMotion(t / 1000);
                onProgress?.(Math.min(1, t / duration));
                if (t >= duration) return resolve();
                requestAnimationFrame(tick);
            };
            tick();
        });
        recorder.stop();
        await done;
        this.renderer.setPixelRatio(prevRatio);
        this.resize();
        this._recording = false;
        this.settings.motion = prevMotion;
        download(new Blob(chunks, { type: mime || "video/webm" }), `${this.data.name}_turntable.${ext}`);
    }

    async export360Frames(frameCount = 36, longSide = 1200, onProgress) {
        if (!this.data) return;
        const [w, h] = this._exportSize(longSide);
        const files = [];
        for (let i = 0; i < frameCount; i++) {
            const angle = (i / frameCount) * Math.PI * 2;
            const canvas = this._renderAt(w, h, angle);
            const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
            files.push({ name: `${this.data.name}_${String(i + 1).padStart(3, "0")}.png`, data: new Uint8Array(await blob.arrayBuffer()) });
            onProgress?.((i + 1) / frameCount);
            await new Promise((r) => setTimeout(r, 0));
        }
        download(makeZip(files), `${this.data.name}_360_${frameCount}frames.zip`);
    }
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("Could not decode image"));
        img.src = src;
    });
}

function imageToCanvas(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    c.getContext("2d").drawImage(img, 0, 0);
    return c;
}

function readPixels(img, w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d", { willReadFrequently: true, colorSpace: "srgb" });
    ctx.drawImage(img, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h).data;
}

function sampleBilinear(arr, w, h, u, v) {
    const x = Math.min(w - 1, Math.max(0, u * (w - 1)));
    const y = Math.min(h - 1, Math.max(0, v * (h - 1)));
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
    const fx = x - x0, fy = y - y0;
    const a = arr[y0 * w + x0], b = arr[y0 * w + x1];
    const c = arr[y1 * w + x0], d = arr[y1 * w + x1];
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

function blurGrid(grid, w, h) {
    const out = new Float32Array(grid.length);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            let sum = 0, n = 0;
            for (let dy = -1; dy <= 1; dy++) {
                const yy = y + dy;
                if (yy < 0 || yy >= h) continue;
                for (let dx = -1; dx <= 1; dx++) {
                    const xx = x + dx;
                    if (xx < 0 || xx >= w) continue;
                    sum += grid[yy * w + xx];
                    n++;
                }
            }
            // keep the silhouette edge at zero so front and back still meet
            out[y * w + x] = grid[y * w + x] === 0 ? 0 : sum / n;
        }
    }
    return out;
}

function flipWinding(geometry) {
    const index = geometry.index;
    if (!index) return;
    const arr = index.array;
    for (let i = 0; i < arr.length; i += 3) {
        const t = arr[i + 1];
        arr[i + 1] = arr[i + 2];
        arr[i + 2] = t;
    }
    index.needsUpdate = true;
}

function download(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
}

// Minimal ZIP writer (stored, no compression - PNGs are already compressed).
const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

function crc32(data) {
    let c = 0xffffffff;
    for (let i = 0; i < data.length; i++) c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

export function makeZip(files) {
    const enc = new TextEncoder();
    const parts = [];
    const central = [];
    let offset = 0;
    for (const file of files) {
        const name = enc.encode(file.name);
        const crc = crc32(file.data);
        const local = new DataView(new ArrayBuffer(30));
        local.setUint32(0, 0x04034b50, true);
        local.setUint16(4, 20, true);
        local.setUint16(8, 0, true);
        local.setUint32(14, crc, true);
        local.setUint32(18, file.data.length, true);
        local.setUint32(22, file.data.length, true);
        local.setUint16(26, name.length, true);
        parts.push(new Uint8Array(local.buffer), name, file.data);

        const cen = new DataView(new ArrayBuffer(46));
        cen.setUint32(0, 0x02014b50, true);
        cen.setUint16(4, 20, true);
        cen.setUint16(6, 20, true);
        cen.setUint32(16, crc, true);
        cen.setUint32(20, file.data.length, true);
        cen.setUint32(24, file.data.length, true);
        cen.setUint16(28, name.length, true);
        cen.setUint32(42, offset, true);
        central.push(new Uint8Array(cen.buffer), name);
        offset += 30 + name.length + file.data.length;
    }
    const centralSize = central.reduce((n, p) => n + p.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, centralSize, true);
    end.setUint32(16, offset, true);
    return new Blob([...parts, ...central, new Uint8Array(end.buffer)], { type: "application/zip" });
}

// ----------------------------------------------------------------
// Page wiring
// ----------------------------------------------------------------
export function initViewerSection() {
    const stage = $("viewerStage");
    if (!stage) return null;
    let viewer;
    try {
        viewer = new JewelleryViewer(stage);
    } catch (err) {
        $("viewerStatus").textContent = "WebGL is not available in this browser.";
        console.error(err);
        return null;
    }
    let currentFile = null;
    let currentIsCutout = false;

    const status = (text, busy = false) => {
        $("viewerStatus").textContent = text || "";
        $("viewerOverlay").classList.toggle("is-visible", busy);
        if (busy) $("viewerOverlayText").textContent = text;
    };

    async function build(file, { isCutout = false } = {}) {
        currentFile = file;
        currentIsCutout = isCutout;
        const fd = new FormData();
        fd.append("image", file, file.name);
        fd.append("depth_mode", $("v3dDepthMode").value);
        fd.append("roundness", $("v3dRoundness").value);
        fd.append("remove_bg", isCutout ? "no" : "auto");
        fd.append("csrfmiddlewaretoken", document.querySelector("[name=csrfmiddlewaretoken]")?.value || "");
        status(isCutout ? "Building 3D model…" : "Detecting jewellery and building 3D model…", true);
        try {
            const res = await fetch("/make-3d/", { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || "Failed to build 3D model");
            await viewer.load(data);
            $("viewerEmpty").style.display = "none";
            $("v3dName").textContent = `${data.name} · ${data.width}×${data.height} · depth: ${data.depth_mode}`;
            const warn = $("v3dWarnings");
            warn.innerHTML = "";
            (data.warnings || []).forEach((w) => {
                const li = document.createElement("li");
                li.textContent = w;
                warn.appendChild(li);
            });
            document.querySelectorAll("[data-needs-model]").forEach((el) => (el.disabled = false));
            status("");
        } catch (err) {
            status("");
            $("viewerStatus").textContent = err.message;
        }
    }

    // source inputs
    $("v3dInput").addEventListener("change", function () {
        if (this.files[0]) build(this.files[0]);
        this.value = "";
    });
    $("v3dRebuild").addEventListener("click", () => currentFile && build(currentFile, { isCutout: currentIsCutout }));

    // live controls
    const bindRange = (id, key, fmt = (v) => v) => {
        const el = $(id);
        const out = $(id + "Val");
        const apply = () => {
            const v = parseFloat(el.value);
            if (out) out.textContent = fmt(v);
            viewer.set(key, v);
        };
        el.addEventListener("input", apply);
        if (out) out.textContent = fmt(parseFloat(el.value));
    };
    bindRange("v3dDepth", "depth", (v) => `${v}%`);
    bindRange("v3dDetail", "detail", (v) => `${v}%`);
    bindRange("v3dSmooth", "smooth");
    bindRange("v3dLight", "light", (v) => v.toFixed(2));
    bindRange("v3dExposure", "exposure", (v) => v.toFixed(2));
    bindRange("v3dLoop", "loopSeconds", (v) => `${v}s`);
    bindRange("v3dSwing", "swingDegrees", (v) => `${v}°`);
    $("v3dRoundnessVal").textContent = $("v3dRoundness").value;
    $("v3dRoundness").addEventListener("input", (e) => ($("v3dRoundnessVal").textContent = e.target.value));

    const bindSelect = (id, key, parse = (v) => v) =>
        $(id).addEventListener("change", (e) => viewer.set(key, parse(e.target.value)));
    bindSelect("v3dMaterial", "material");
    bindSelect("v3dMotion", "motion");
    bindSelect("v3dBack", "back");
    bindSelect("v3dResolution", "resolution", (v) => parseInt(v, 10));
    $("v3dShadow").addEventListener("change", (e) => viewer.set("shadow", e.target.checked));
    $("v3dWire").addEventListener("change", (e) => viewer.set("wireframe", e.target.checked));
    $("v3dTransparent").addEventListener("change", (e) => viewer.set("transparent", e.target.checked));
    $("v3dBg").addEventListener("input", (e) => viewer.set("background", e.target.value));
    $("v3dReset").addEventListener("click", () => viewer.resetView());

    // exports
    const busy = async (button, fn) => {
        const label = button.textContent;
        button.disabled = true;
        try {
            await fn((p) => (button.textContent = `${Math.round(p * 100)}%`));
        } catch (err) {
            alert(err.message || err);
        } finally {
            button.textContent = label;
            button.disabled = false;
        }
    };
    $("v3dPng").addEventListener("click", (e) => busy(e.currentTarget, () => viewer.snapshotPNG(2000)));
    $("v3dGlb").addEventListener("click", (e) => busy(e.currentTarget, () => viewer.exportGLB()));
    $("v3dVideo").addEventListener("click", (e) => busy(e.currentTarget, (p) => viewer.recordVideo(p)));
    $("v3dFrames").addEventListener("click", (e) =>
        busy(e.currentTarget, (p) => viewer.export360Frames(parseInt($("v3dFrameCount").value, 10), 1200, p)));

    // drag & drop onto the stage
    stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("is-drag"); });
    stage.addEventListener("dragleave", () => stage.classList.remove("is-drag"));
    stage.addEventListener("drop", (e) => {
        e.preventDefault();
        stage.classList.remove("is-drag");
        const file = [...e.dataTransfer.files].find((f) => f.type.startsWith("image/"));
        if (file) build(file);
    });

    // "Open in 3D" from the background-removal results
    window.addEventListener("open-in-3d", async (e) => {
        const { url, base64, name } = e.detail || {};
        try {
            let blob;
            if (url) blob = await (await fetch(url)).blob();
            else blob = await (await fetch("data:image/png;base64," + base64)).blob();
            build(new File([blob], (name || "jewellery") + ".png", { type: "image/png" }), { isCutout: true });
        } catch (err) {
            $("viewerStatus").textContent = "Could not open that image: " + err.message;
        }
    });

    fetch("/depth-status/").then((r) => r.json()).then((d) => {
        $("v3dAiHint").textContent = d.ai_depth_ready
            ? "AI depth model ready."
            : "AI depth model downloads (~100 MB) on first use; Shape mode works offline.";
    }).catch(() => {});

    window.__jewelleryViewer = viewer;
    return viewer;
}
