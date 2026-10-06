// Lazy 3D viewer for the loop-closed purple-hat cat splat.
// Spark (MIT, sparkjs.dev) renders the Gaussian splat on top of three.js; both load from jsDelivr.
// Nothing here needs SharedArrayBuffer or special headers, so it runs on plain GitHub Pages.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

// Rigid transform from the reconstruction's own (COLMAP) frame to an orbit-aligned frame:
// origin = the point every training camera looks at, +Y = the orbit plane's normal,
// +Z = towards the first frame's camera. Computed from the 124 loop-closed camera poses.
const ALIGN = {
  quaternion: [0.6628832178439559, -0.25604857350749294, -0.317239902915609, -0.6279998499233225],
  translate: [-0.0036920278246615184, -0.16422868179730987, -0.04488609542327237],
};
// First training camera in that frame (radius 0.886, about 0.5 degrees above the focus point).
const START = new THREE.Vector3(0, 0.0077, 0.8863);
// The pinhole fitted by structure-from-motion: f = 1417 px at 768 px height -> 30.3 deg vertical.
const FOV_DEG = 30.3;

export async function mountViewer(container, { url, onProgress, autoRotate = true, onRotate } = {}) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1418);
  const camera = new THREE.PerspectiveCamera(FOV_DEG, 4 / 3, 0.01, 200);
  camera.position.copy(START);

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const mesh = new SplatMesh({ url, onProgress });
  mesh.quaternion.set(...ALIGN.quaternion);
  mesh.position.set(...ALIGN.translate);
  scene.add(mesh);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.minDistance = 0.35;
  controls.maxDistance = 2.5;
  const setAuto = (on) => { controls.autoRotate = on; if (onRotate) onRotate(on); };
  setAuto(autoRotate);
  controls.autoRotateSpeed = -1.2;
  controls.update();
  const stop = () => setAuto(false);
  renderer.domElement.addEventListener('pointerdown', stop);
  renderer.domElement.addEventListener('wheel', stop, { passive: true });

  function resize() {
    const w = container.clientWidth;
    const h = Math.max(220, Math.round(w * 0.75));
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = 'auto';
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  new ResizeObserver(resize).observe(container);

  const loop = () => { controls.update(); renderer.render(scene, camera); };
  let running = false;
  const setRunning = (on) => { if (on !== running) { running = on; renderer.setAnimationLoop(on ? loop : null); } };
  setRunning(true);
  // Stop drawing while the viewer is scrolled out of sight (saves battery on phones).
  new IntersectionObserver((entries) => setRunning(entries[0].isIntersecting)).observe(container);

  await mesh.initialized;

  return {
    numSplats: mesh.packedSplats?.numSplats ?? null,
    // Rotation toggle (the Rotate button): on/off, or flip when called with no argument.
    setRotate(on) { setAuto(on === undefined ? !controls.autoRotate : !!on); return controls.autoRotate; },
    reset() { camera.position.copy(START); controls.target.set(0, 0, 0); controls.update(); },
    // Place the camera on the training orbit at a given azimuth (degrees, 0 = first frame).
    setAzimuth(deg) {
      setAuto(false);
      const a = (deg * Math.PI) / 180;
      const r = Math.hypot(START.x, START.z);
      camera.position.set(r * Math.sin(a), START.y, r * Math.cos(a));
      controls.target.set(0, 0, 0);
      controls.update();
    },
    // Glide along the training orbit to an azimuth over ~0.8 s. A smooth move gives the splat
    // sorter time to keep up; an instant jump of 180 degrees can show a stale, ghosted frame.
    flyTo(deg) {
      setAuto(false);
      const r = Math.hypot(START.x, START.z);
      const from = Math.atan2(camera.position.x, camera.position.z);
      let to = (deg * Math.PI) / 180;
      while (to - from > Math.PI) to -= 2 * Math.PI;
      while (to - from < -Math.PI) to += 2 * Math.PI;
      const t0 = performance.now();
      const step = (now) => {
        const u = Math.min(1, (now - t0) / 800);
        const e = u * u * (3 - 2 * u);
        const a = from + (to - from) * e;
        camera.position.set(r * Math.sin(a), START.y, r * Math.cos(a));
        controls.target.set(0, 0, 0);
        controls.update();
        if (u < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    },
    // For offline turntable capture: stop the loop, render, return a PNG data URL.
    pause() { setRunning(false); },
    capture() { renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); },
  };
}
