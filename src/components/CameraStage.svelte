<script>
  import { camera, isFocused } from '../core/camera/cameraStore.js';
</script>

<div class="camera-viewport">
  <div class="vignette" class:active={isFocused($camera)} aria-hidden="true"></div>
  <div
    class="camera-plane"
    style="transform: translate3d({$camera.x}px, {$camera.y}px, 0) scale({$camera.scale});"
  >
    <slot />
  </div>
</div>

<style>
  .camera-viewport {
    width: 100%;
    height: 100%;
    overflow: hidden;
    position: relative;
  }
  .camera-plane {
    width: 100%;
    height: 100%;
    will-change: transform;
    transform-origin: 50% 45%;
  }
  .vignette {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    background: radial-gradient(
      ellipse at center,
      transparent 55%,
      rgba(6, 7, 11, 0.55) 100%
    );
    opacity: 0;
    transition: opacity 0.5s ease;
  }
  .vignette.active { opacity: 1; }

  @media (prefers-reduced-motion: reduce) {
    .camera-plane { transition: none; }
  }
</style>
